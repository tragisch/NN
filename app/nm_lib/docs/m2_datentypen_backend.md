# M2: Datentypen und Backend-Schnittstelle

Stand: Schritte 1 bis 3 sind umgesetzt (siehe unten), 4 und 5 warten auf mMatrix. Ziel laut Roadmap: Datentyp
pro Wert in Graph und Engine, bf16 für Training mit f32-Mastergewichten,
Backend (CPU/GPU) als Laufzeitwahl mit gerätegebundenen Tensoren. Fertig, wenn
ein MLP und ein CNN in bf16 trainiert sind und die Genauigkeit im Rahmen von
f32 liegt.

## Befunde

- **nm_lib:** Der Graph kennt nur f32 (`NmIrValue` hat keinen Datentyp, der
  Importer lehnt alles außer `float32` ab). Training verlangt f32-Gewichte.
  bf16 gibt es nur im alten `.nms`-Format (beim Laden nach f32 gewandelt).
- **mMatrix:** `st` hat `ST_DTYPE_BF16` und Umwandlungen f32 und bf16; `sm`
  (Matrizen, GEMM) rechnet nur f32. Conv rechnet in bf16 über Promotion.
- **GraphIR:** bf16 gilt für den ganzen Graphen. `MatMul` verlangt gleiche
  Datentypen für beide Eingänge und die Ausgabe, ein bf16-Gewicht mit
  f32-Aktivierung ist ungültig; dafür gibt es die Operation `Cast`. Der
  Referenz-Executor rechnet nur f32, es gibt also keine Referenzausgaben für
  bf16-Graphen.
- **Backend-Wahl:** `NmEngineCConfig.backend` kennt `AUTO` und `DEFAULT`, beide
  führen zur selben Operationstabelle (`NM_BACKEND_DEFAULT_OPS`). Die Auswahl
  zwischen CPU und GPU passiert in mMatrix zur Compile-Zeit
  (`-DUSE_ACCELERATE_MPS`).

## Schritte

1. **Gewichte als `.girw` (umgesetzt):** `nm_export_graphir` kann die Gewichte in
   ein geprüftes Paket schreiben (`weights_path`), `nm_network_save` nutzt es
   (`model.graphir.json` + `weights.girw`). GraphIR prüft und löst das Paket
   auf (`check`, `rust --weights`). Das schließt die Abweichung aus M1. Dieser
   Schritt ist die Grundlage, weil bf16-Gewichte nur über `.girw` laufen.
2. **Datentyp im Graphen:** `NmIrValue` bekommt `dtype`; Import und Export
   lesen und schreiben `float16`/`bfloat16`. Ausführung zunächst über
   Promotion (Gewichte und Aktivierungen nach f32, rechnen, Ausgabe
   zurückrunden). Das liefert bf16-Modelle aus GraphIR oder PyTorch, ist aber
   keine echte bf16-Rechnung. Das Profil bekommt die Datentypen mit dem
   Vermerk "promoted".
3. **Referenz in GraphIR:** Der Referenz-Executor rundet nach jeder Operation
   auf bf16 (Emulation). Erst damit gibt es Korpusfälle, gegen die nm_lib bf16
   prüfen kann. Das ist eine GraphIR-Funktion und nützt allen Backends.
4. **Engine rechnet echtes bf16:** Sobald mMatrix bf16-GEMM und elementweise
   bf16-Kerne hat (siehe unten). Training mit f32-Mastergewichten: Vorwärts und
   Rückwärts in bf16, der Optimizer hält f32-Gewichte und Zustand und schreibt
   die bf16-Kopie zurück. Gradienten-Check mit angepasster Toleranz.
5. **Backend zur Laufzeit:** Die Operationstabelle wird zu einer Tabelle pro
   Gerät, Tensoren gehören zu einem Gerät, die Engine wählt beim Prepare.
   Zunächst nur CPU fertig, GPU folgt in M6.

## Umgesetzt (Schritte 1 bis 3)

- **Schritt 2, Datentyp im Graphen:** `NmIrValue.dtype` (f32, bf16, f16). Der
  Importer liest `bfloat16`/`float16`-Graphen mit Gewichten im `.girw`-Paket und
  weitet sie exakt nach f32; ein Graph hat einen Float-Typ, andere Typen
  (`float64`, Integer) sind `NM_STATUS_UNSUPPORTED_TYPE`. Die Engine rechnet in
  f32 (Promotion), die Ausgabe ist nicht auf bf16 gerundet. Der Export schreibt
  den Typ und die Gewichte wieder in bf16/f16 (nur mit `.girw`, Inline-JSON
  kennt nur f32). `fit` auf einem 16-Bit-Graphen gibt `UNSUPPORTED_TYPE`, denn
  Training braucht echte 16-Bit-Kerne (Schritt 4). Das Profil führt beide Typen
  unter `promoted_dtypes`.
- **Schritt 3, Referenz in GraphIR:** Der Executor emuliert bf16 und f16
  (f32-Rechnung, Runden jeder Ausgabe, `halfprec.rs`). Der Korpus hat je einen
  bf16- und f16-Fall mit eigener Toleranz (0,05 und 0,01) und einen
  unsupported-Fall für `float64`. Die C-Umwandlungen (`nm_halfprec.h`) sind für
  alle 65 536 Muster (Hin- und Rückweg) und mit denselben Randwerten wie in GraphIRs Test geprüft.
- **Grenze:** Ein Modell mit `Cast` oder gemischten Typen lehnt nm_lib ab.

## Was nm_lib von mMatrix erwartet

Alles in `sm` (Matrizen) oder `st` (Tensoren), je nach Ort der Funktion heute.
Namen sind Vorschläge.

| Baustein | Erwartung | Zweck |
| --- | --- | --- |
| Datentyp an Matrizen | `FloatMatrix` bekommt wie `FloatTensor` einen `dtype` (f32, bf16) | Dense-Pfad ohne Umweg über Tensoren |
| GEMM | bf16 x bf16 mit f32-Akkumulation, Ausgabe bf16 oder f32; Transponierung A/B wie heute; optional Bias und ReLU fusioniert | Dense vorwärts, Rückwärts (dW, dX) |
| Umwandlung | f32 nach bf16 und zurück, vektorisiert (NEON), mit Rundung auf nächsten Wert (round-to-nearest-even) und `NaN`/`Inf`-Behandlung, auch "in place" in vorreservierten Puffern | Mastergewichte, Optimizer |
| Elementweise | ReLU, Sigmoid, Tanh, Softmax und ihre Gradienten, Addition, Skalierung für bf16 mit f32-Zwischenwerten | Aktivierungen |
| Reduktion | Zeilensummen und Verlustsummen mit f32-Akkumulator | Bias-Gradient, Loss |
| Conv und Pooling | bf16-Vorwärts und -Rückwärts ohne Promotion-Kopie | CNN-Training |
| Gerät | ein Gerätebegriff (`CPU`, `GPU`) an Tensoren und Matrizen, plus Funktion "Daten für Gerät X sichtbar machen" (bei Shared Memory: nur Synchronisation) | Backend zur Laufzeit |
| Synchronisation | ein expliziter Aufruf, der wartet, bis GPU-Arbeit auf einem Tensor fertig ist, und einer, der Arbeit einreiht | GPU-Training in M6 |
| Fähigkeiten | eine Abfrage zur Laufzeit: welche Datentypen und Geräte gibt es, welche Kerne sind schnell (z. B. `bf16_gemm`) | Auswahl durch die Engine |
| Determinismus | ein Schalter für feste Reihenfolge der Summen | reproduzierbare Läufe |

Reihenfolge, die nm_lib am meisten hilft: Umwandlung, bf16-GEMM, elementweise
Kerne, danach Conv und Pooling, zuletzt Gerätebegriff und Synchronisation.
Die bf16-Teile in mMatrix werden aktiv weiterentwickelt; dieses Dokument ist der Wunsch von
nm_lib, keine Vorgabe.

## Offene Entscheidungen

| Frage | Vorschlag |
| --- | --- |
| Promotion als erster bf16-Weg akzeptabel, obwohl die Rechnung nicht in bf16 läuft? | ja, mit klarer Kennzeichnung im Profil, damit niemand bf16-Genauigkeit erwartet |
| Wo liegt der Datentyp bei Aktivierungen im Graphen? | pro Wert, wie in GraphIR; gemischte Graphen nur über `Cast` |
| Lernrate und Verlustskalierung bei bf16 | zuerst ohne Verlustskalierung (bf16 hat den Bereich von f32), später prüfen |
| Soll Schritt 3 (Referenz-Emulation) in GraphIR jetzt beginnen? | ja, parallel zu Schritt 2 |
