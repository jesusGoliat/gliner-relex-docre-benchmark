# Atribución y licencias de terceros

## Re-DocRED (datos incluidos en `data/subsets/redocred/`)

Los ficheros `train_N{10,50}_s{0..4}.json` son subconjuntos de documentos del split de entrenamiento revisado de **Re-DocRED**
(Tan, Xu, Bing, Ng, Aljunied. *Revisiting DocRED – Addressing the False Negative Problem in Relation Extraction*. EMNLP 2022;
https://github.com/tonytan48/Re-DocRED, commit `ccfb54f5ddf5836027c87badda10f6dfc56efaac`). Re-DocRED se distribuye con la licencia MIT
cuyo texto se reproduce a continuación, como exige la propia licencia.

Re-DocRED deriva de DocRED, cuyos textos proceden de Wikipedia; la licencia de ese texto subyacente no se ha verificado en este repositorio.

Los datos restantes (`dev_revised.json`, `test_revised.json`, `train_revised.json`) y la tabla dominio/rango de DocRE-StructuralConsistency
(https://github.com/mntlra/DocRE-StructuralConsistency, MIT) **no se redistribuyen**: se descargan con `data/download.sh`.

### Texto de la licencia de Re-DocRED

MIT License

Copyright (c) 2023 tonytan48

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
