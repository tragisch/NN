/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Directory transport: one file per message, written atomically. */
#include "nm_connector.h"
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define KEEP_NEWEST 3
#define NAME_PREFIX "weights."
#define NAME_SUFFIX ".nmws"
#define NAME_WIDTH 20

typedef struct {
  char *directory;
  char last_seen[64]; /* receiver: name of the newest file already handed out */
} FileState;

static void free_buffer(void *buffer) { free(buffer); }

static int cmp_names(const void *a, const void *b) {
  return strcmp(*(char *const *)a, *(char *const *)b);
}

static bool is_message_name(const char *name) {
  size_t length = strlen(name);
  return length == strlen(NAME_PREFIX) + NAME_WIDTH + strlen(NAME_SUFFIX) &&
         !strncmp(name, NAME_PREFIX, strlen(NAME_PREFIX)) &&
         !strcmp(name + length - strlen(NAME_SUFFIX), NAME_SUFFIX);
}

/* Sorted names of message files; caller frees the array and its strings. */
static size_t list_messages(const char *directory, char ***names) {
  *names = NULL;
  DIR *dir = opendir(directory);
  if (!dir)
    return 0;
  size_t count = 0, capacity = 0;
  struct dirent *entry;
  while ((entry = readdir(dir))) {
    if (!is_message_name(entry->d_name))
      continue;
    if (count == capacity) {
      capacity = capacity ? capacity * 2 : 8;
      char **grown = realloc(*names, capacity * sizeof(char *));
      if (!grown)
        break;
      *names = grown;
    }
    (*names)[count++] = strdup(entry->d_name);
  }
  closedir(dir);
  if (count)
    qsort(*names, count, sizeof(char *), cmp_names);
  return count;
}

static void free_names(char **names, size_t count) {
  for (size_t i = 0; i < count; ++i)
    free(names[i]);
  free(names);
}

static NmStatus file_loan(void *context, size_t size, void **buffer) {
  (void)context;
  void *memory = NULL;
  if (size == 0 || posix_memalign(&memory, NM_WEIGHTSET_ALIGNMENT,
                                  (size + 63) / 64 * 64) != 0)
    return NM_STATUS_ALLOC_FAILED;
  *buffer = memory;
  return NM_STATUS_OK;
}

static void file_discard(void *context, void *buffer) {
  (void)context;
  free(buffer);
}

static NmStatus file_send(void *context, void *buffer, size_t size) {
  FileState *state = context;
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  char name[128], temporary[1024], final_path[1024];
  snprintf(name, sizeof(name), NAME_PREFIX "%020llu" NAME_SUFFIX,
           (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec);
  snprintf(temporary, sizeof(temporary), "%s/.tmp.%d.%s", state->directory,
           (int)getpid(), name);
  snprintf(final_path, sizeof(final_path), "%s/%s", state->directory, name);

  FILE *file = fopen(temporary, "wb");
  NmStatus status = NM_STATUS_IO_ERROR;
  if (file) {
    bool ok = fwrite(buffer, 1, size, file) == size;
    ok = (fclose(file) == 0) && ok;
    if (ok && rename(temporary, final_path) == 0)
      status = NM_STATUS_OK;
    else
      remove(temporary);
  }
  free(buffer);
  if (status != NM_STATUS_OK)
    return status;

  char **names = NULL;
  size_t count = list_messages(state->directory, &names);
  for (size_t i = 0; i + KEEP_NEWEST < count; ++i) {
    char old[1024];
    snprintf(old, sizeof(old), "%s/%s", state->directory, names[i]);
    remove(old);
  }
  free_names(names, count);
  return NM_STATUS_OK;
}

static NmStatus file_receive(void *context, const void **message, size_t *size,
                             NmWeightRelease *release, void **release_context) {
  FileState *state = context;
  *message = NULL;
  char **names = NULL;
  size_t count = list_messages(state->directory, &names);
  for (size_t i = 0; i < count; ++i) {
    if (strcmp(names[i], state->last_seen) <= 0)
      continue;
    snprintf(state->last_seen, sizeof(state->last_seen), "%s", names[i]);
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", state->directory, names[i]);
    FILE *file = fopen(path, "rb");
    if (!file)
      continue; /* removed meanwhile */
    long length = 0;
    void *memory = NULL;
    if (fseek(file, 0, SEEK_END) == 0 && (length = ftell(file)) > 0 &&
        fseek(file, 0, SEEK_SET) == 0 &&
        posix_memalign(&memory, NM_WEIGHTSET_ALIGNMENT,
                       ((size_t)length + 63) / 64 * 64) == 0) {
      if (fread(memory, 1, (size_t)length, file) == (size_t)length) {
        fclose(file);
        free_names(names, count);
        *message = memory;
        *size = (size_t)length;
        *release = free_buffer;
        *release_context = memory;
        return NM_STATUS_OK;
      }
      free(memory);
    }
    fclose(file);
  }
  free_names(names, count);
  return NM_STATUS_OK;
}

static void file_close(void *context) {
  FileState *state = context;
  free(state->directory);
  free(state);
}

static NmStatus file_open(const char *directory, NmTransport *transport,
                          bool sender) {
  if (!directory || !transport)
    return NM_STATUS_INVALID_ARGUMENT;
  if (mkdir(directory, 0755) != 0 && errno != EEXIST)
    return NM_STATUS_IO_ERROR;
  FileState *state = calloc(1, sizeof(*state));
  if (!state)
    return NM_STATUS_ALLOC_FAILED;
  state->directory = strdup(directory);
  if (!state->directory) {
    free(state);
    return NM_STATUS_ALLOC_FAILED;
  }
  if (sender) {
    *transport = (NmTransport){.context = state,
                               .loan = file_loan,
                               .send = file_send,
                               .discard = file_discard,
                               .close = file_close};
  } else {
    *transport = (NmTransport){.context = state,
                               .receive = file_receive,
                               .close = file_close};
  }
  return NM_STATUS_OK;
}

NmStatus nm_transport_file_sender(const char *directory, NmTransport *sender) {
  return file_open(directory, sender, true);
}

NmStatus nm_transport_file_receiver(const char *directory, NmTransport *receiver) {
  return file_open(directory, receiver, false);
}
