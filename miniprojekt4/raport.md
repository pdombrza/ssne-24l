## Miniprojekt 4
Mateusz Kiełbus, Paweł Dombrzalski

### Temat
Tematem zadania było stworzenie modelu generatywnego, generującego obrazki znaków drogowych.

### Struktura projektu
W pliku analysis.ipynb znajduje się wstępne przejrzenie obrazków. Sprawdziliśmy jak wyglądają same obrazy. W plikach .py znajdują się różne testowane przez nas modele.

### Wykonane eksperymenty
W ramach eksperymentów sprawdzaliśmy jak różne rozmiary sieci oraz architektury wpłyną na ostateczny wynik. Porównaliśmy VAE i GANy. Do porównania modeli zastosowaliśmy Frechet Inception Distance - mierzyliśmy odległość między wygenerowanymi obrazami a wydzielonym fragmentem zbioru testowego.

### VAE
Już podstawowy model, z dwoma warstwami liniowymi, osiągnął całkiem dobry wynik - wartość FID ok. 220. Dodanie większej ilości warstw liniowych nie miało znaczącego wpływu na jakość generowanych obrazów - jedynie wydłużyło proces uczenia, uzyskanie podobnych wyników wymagało zwiększenia liczby epok. Z kolei dodanie warstw konwolucyjnych poskutkowało polepszeniem jakości modelu - wygenerowane obrazki wyglądały lepiej, a osiągnięta wartość FID wynosiła ok. 110. Zwiększenie przestrzeni ukrytej również poprawiło generowane znaki drogowe.


### GAN


### Ostateczny model
Ostatecznie zdecydowaliśmy się zastosować model VAE z warstwami konwolucyjnymi. Enkoder składa się z 4 warstw konwolucyjnych i 2 liniowych, między warstwami stosujemy Batch Normalization. Funkcją aktywacji jest leaky ReLU. Analogicznie w dekoderze stosujemy 2 warstwy liniowe i 4 dekonwolucyjne. Warstwa ukryta VAE ma rozmiar 256. Przy treningu zastosowaliśmy transformację `RandomHorizontalFlip`.