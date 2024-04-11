## Miniprojekt 2
Mateusz Kiełbus, Paweł Dombrzalski

### Temat
Tematem zadania było przewidzenie ceny mieszkania - czy jest ona mniejsza niż 100tys, znajduje się w przedziale od 100 do 350 tys, czy jest większa od 350tys. Z analizy danych można było zauważyć, że w danych treningowych klasy są niezbalansowane - mieszkań z przedziału cenowego od 100 do 350 tys. jest więcej niż pozostałych. Znajduje się też wiele atrybutów dyskretnych, które wymagają obsłużenia.

### Struktura projektu
W pliku analysis.ipynb znajduje się krótka analiza dostępnych danych. W pliku model.py znajduje się ostatecznie utworzony model. W pozostałych plikach znajdują się testowane modele. W katalogu results znajdują się wyniki testów poszczególnych modeli.

### Wykonane eksperymenty
* Sprawdzenie wpływu warstw BatchNormalization i Dropout na otrzymany model
* Sprawdzenie wpływu regularyzacji zmian wag na końcowy model
* Zastosowanie embeddingu na kolumnach dyskretnych
* Zbadanie różnych rozmiarów modelu oraz ilości epok
Dla każdego modelu stosowaliśmy funkcję aktywacji ReLU i optimizer SGD.

### Wyniki eksperymentów
Ostatecznie okazało się że dodanie warstw BatchNormalization i Dropout wpłynęły pozytywnie na wyniki modelu. Dodanie regularyzacji też miało pozytywny wpływ. Z kolei zastosowanie embeddingu nie poprawiło znacząco wyników. Natomiast zbyt duże wartości parametru warstwy Dropout i parametru weight decay wpłynęły negatywnie na uzyskane wyniki. Bez zastosowania warstw Batch Normalization i Dropout sieci o bardziej rozbudowanej strukturze - o większej ilości warstw i neuronów na warstwę - uczyły się gorzej. Być może wynikało to ze zbyt małej ilości epok. Bardziej skomplikowane sieci też uczyły się wolniej. Modele uzyskiwały trochę gorsze wyniki dla rzadziej występujących w danych treningowych klas. Wyniki dla poszczególnych klas były trochę bardziej zrównoważone dla modeli z warstwami Dropout i wartością parametru równą 0.4, niż w pozostałych przypadkach.


### Ostateczny model
Ostatecznie postanowiliśmy zastosować model z wartstwą BatchNormalization i Dropout, znajdujący się w pliku model.py.