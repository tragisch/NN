# M1: Anwender-API von nm_lib, Entwurf zur Prüfung

Stand (Oktober 2026): umgesetzt in `include/nm_api.h` und `src/nm_api.c`. Abweichungen vom Entwurf: Handle heißt `NmNetwork` (`NmModel` ist im alten `nm.h` belegt), `predict` allokiert die Ausgabe, `evaluate` bekommt den Loss, Gewichte liegen seit M2 in `weights.girw` neben dem JSON (`nm_network_export_graphir` schreibt sie weiter inline). Ziel: ein kleines, stabiles C-Interface, das in
C, Swift (GraphGUI) und später über einen Python-Connector gleich aussieht.

## Ziele

1. Ein Header, `nm_api.h`, mit allem, was ein Anwender braucht. `nm.h`
   (Legacy) und die `nm_engine_c`-Funktionen bleiben, werden aber nicht mehr
   gezeigt.
2. Ein Modell, ein Handle: bauen, trainieren, vorhersagen, speichern.
3. Kein `FloatMatrix` in der öffentlichen API. Daten gehen als `NmTensor` rein
   und raus (siehe unten), damit Swift und DLPack ohne Umweg passen.
4. Klassen von Problemen: ANN, Autoencoder, CNN heute; RNN (M4) ist in der
   API schon vorgesehen, aber noch nicht implementiert.
5. Fehler sind lesbar: `NmStatus` plus `nm_last_error()`.

## Bausteine

```c
/* Daten: geliehene Sicht, nm_lib kopiert nicht und gibt nicht frei. */
typedef struct {
  float *data;          /* row-major, float32 */
  size_t ndim;          /* 1..4 */
  size_t dims[4];       /* dims[0] = Batch (oder Zeitschritte bei Sequenzen) */
} NmTensor;

typedef struct NmNetwork NmNetwork;            /* ein Handle für alles */
typedef struct NmSpec  NmSpec;             /* Modellbeschreibung, wiederverwendbar */

/* 1. Beschreiben */
NmSpec *nm_spec_new(NmShape input);
NmStatus nm_spec_dense(NmSpec *, size_t units, NmActivation);
NmStatus nm_spec_conv2d(NmSpec *, size_t filters, size_t kernel, size_t stride, size_t pad, NmActivation);
NmStatus nm_spec_maxpool2d(NmSpec *, size_t kernel, size_t stride);
NmStatus nm_spec_flatten(NmSpec *);
/* M4: nm_spec_rnn / nm_spec_gru, Eingabe [T, F] */
void nm_spec_free(NmSpec *);

/* 2. Bauen: prüft, initialisiert Gewichte (Seed!), bereitet die Engine vor. */
NmNetwork *nm_network_build(const NmSpec *, const NmBuildOptions *);   /* NULL bei Fehler */
void nm_network_free(NmNetwork *);

/* 3. Trainieren (überwacht und unüberwacht) */
NmStatus nm_network_fit(NmNetwork *, NmTensor x, const NmTensor *y,    /* y == NULL: Autoencoder, Ziel = x */
                      const NmFitOptions *, NmProgressFn, void *user);

/* 4. Anwenden */
NmStatus nm_network_predict(const NmNetwork *, NmTensor x, NmTensor *out);   /* nm_lib allokiert out, Freigabe mit nm_tensor_free */
NmStatus nm_network_evaluate(const NmNetwork *, NmTensor x, NmTensor y, NmLoss loss, NmMetrics *out);   /* Loss wird übergeben, nicht geraten */

/* 4b. Übersicht */
size_t nm_network_parameter_count(const NmNetwork *);
size_t nm_network_summary(const NmNetwork *, char *buffer, size_t size);   /* wie snprintf */

/* 5. Speichern / Laden / Austausch */
NmStatus nm_network_save(const NmNetwork *, const char *dir);          /* .graphir.json + .girw */
NmNetwork *nm_network_load(const char *dir);
NmStatus nm_network_export_graphir(const NmNetwork *, const char *path);
const char *nm_last_error(void);                                   /* thread-lokal */
```

Optionen sind kleine Structs mit `nm_*_options_default()`:

- `NmBuildOptions`: `seed`, `backend`, `deterministic`.
- `NmFitOptions`: `loss` (MSE, Kreuzentropie), `optimizer` (SGD, Momentum, Adam),
  `learning_rate`, `batch_size`, `epochs`, `shuffle`, `validation_split`.
- `NmProgressFn(epoch, loss, val_loss, user)` wird pro Epoche gerufen, Rückgabe
  ungleich 0 bricht ab. Damit zeichnet GraphGUI Kurven und hat einen Stop-Knopf.

## Beispiel

```c
NmSpec *s = nm_spec_new((NmShape){.ndim = 1, .dims = {784}});
nm_spec_dense(s, 128, NM_ACT_RELU);
nm_spec_dense(s, 10, NM_ACT_SOFTMAX);

NmBuildOptions b = nm_build_options_default();
b.seed = 42;
NmNetwork *m = nm_network_build(s, &b);

NmFitOptions f = nm_fit_options_default();
f.loss = NM_LOSS_CROSS_ENTROPY;
f.epochs = 10;
nm_network_fit(m, x_train, &y_train, &f, NULL, NULL);

NmMetrics r;
nm_network_evaluate(m, x_test, y_test, NM_LOSS_CROSS_ENTROPY, &r);   /* r.loss, r.accuracy */
nm_network_save(m, "mnist.nmmodel");
```

Autoencoder: Spec mit Engpass, dann `nm_network_fit(m, x, NULL, ...)`.

## Swift

- `module.modulemap` im Paket, Header mit `NS_SWIFT_NAME`-Annotationen und
  Nullability (`_Nonnull`/`_Nullable`), damit `NmNetwork` als Klasse-ähnlicher Typ
  erscheint. Keine Varargs, keine Funktionsmakros in der API.
- Kein globaler Zustand außer `nm_last_error()` (thread-lokal). Ausnahme in der
  Umsetzung: ein Seed ungleich 0 in `NmBuildOptions` ruft das globale
  `nm_set_seed` auf.
- Der Fortschritts-Callback läuft im Trainingsthread; GraphGUI muss selbst auf
  den Main-Thread wechseln.

## Verhalten und Garantien

- Aufrufe auf einem `NmNetwork` laufen nacheinander (interne Sperre). Ein Netz
  darf von mehreren Threads benutzt werden, aber `predict` läuft nicht parallel;
  dafür ein Netz pro Thread bauen. Der Fortschritts-Callback läuft unter dieser
  Sperre: kein Aufruf auf demselben Netz aus dem Callback (Deadlock).
- Gleicher Seed, gleiche Daten, `deterministic = true`: gleiches Ergebnis.
- Gebundene Gewichte (Connector) bleiben schreibgeschützt; `fit` gibt dann
  `NM_STATUS_INVALID_ARGUMENT` und eine lesbare Meldung zurück.
- Alte API bleibt lauffähig; die neue API ist eine dünne Schicht über
  Frontend, Graph und Engine (`nm_frontend_*`, `nm_engine_c_*`), keine zweite
  Implementierung.

## Offene Entscheidungen (mit Vorschlag)

| Frage | Vorschlag |
| --- | --- |
| `NmSpec` und `NmNetwork` getrennt, oder ein Typ? | getrennt: eine Spec kann mehrfach gebaut werden (Hyperparameter-Läufe, Ensembles) |
| Datentyp der Eingabe | nur float32; bf16 folgt, wenn mMatrix es kann |
| Labels bei Klassifikation | One-Hot-Tensor wie heute; Index-Labels (`int32`) als Ergänzung in M2 |
| Speicherformat | Ordner mit `model.graphir.json` und `weights.girw` (Prüfsummen), kein neues Format |
| Sequenzen | Eingabe `[T, F]` pro Beispiel, Batch zuerst; erst mit M4 aktiv |
| Fehlerbehandlung | `NmStatus` plus `nm_last_error()`, keine Abbrüche in der Bibliothek |

## Umsetzungsschritte nach Freigabe

1. `nm_api.h`/`nm_api.c` als Schicht über Frontend und Engine, `NmTensor` an den
   Grenzen in `FloatMatrix` wrappen (ohne Kopie, wo das Layout passt).
2. Speichern/Laden über GraphIR-Export/-Import und `.girw`.
3. Fortschritts-Callback und Validierungsanteil im Trainer.
4. Swift-Paket mit Modulemap und ein kleiner Beispieltest (MNIST-MLP).
5. Tests: Round-Trip Speichern/Laden, Autoencoder-Konvergenz, Fehlermeldungen.
