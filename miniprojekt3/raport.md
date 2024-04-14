## Miniprojekt 3
Mateusz Kiełbus, Paweł Dombrzalski

### Temat
Tematem zadania było stworzenie klasyfikatora obrazków działającego na 50 różnych klas.

### Struktura projektu
W pliku analysis.ipynb znajduje się wstępne przejrzenie obrazków. Sprawdziliśmy jak wyglądają same obrazy i jakie są ich rozmiary.

### Wykonane eksperymenty
W ramach eksperymentów sprawdzaliśmy jak różne rozmiary sieci wpłyną na ostateczny wynik. Poszczególne testowane struktury znajdują się w plikach `model_structure_test`. Zbadaliśmy też, czy dodanie regularyzacji wpłynie na końcową jakość modelu.


### Ostateczny model
Ostatecznie postanowiliśmy zastosować model z 10 warstwami konwolucyjnymi i 5 warstwami liniowymi oraz 1 warstwą wyjściową. Zastosowaliśmy też augmentację danych treningowych - transformacje `RandomHorizontalFlip`, `RandomHorizontalFlip`, `RandomRotation` i `RandomErasing`. Taki model wykonywał ok. 56% poprawnych predykcji na zbiorze walidacyjnym.