## Miniprojekt 6

Paweł Dombrzalski, Mateusz Kiełbus

### Temat

Tematem zadania było stworzenie modelu klasyfikującego liczbę gwiazdek przyznanych hotelowi

### Struktura projektu

W pliku _analysis.ipynb_ znajduje się wstępna analiza danych. W pliku _bow\_classifier.py_ znajduje sie kod do podstawowego klasyfikatora metodą bag of words. W pliku _distilbert.py_ wykorzystany został model DistilBERT. W pliku _fast\_text.py_ wykorzystaliśmy do klasyfikacji bibliotekę fasttext.

### Wykonane eksperymenty

W danych treningowych klasy są niezbalansowane - aby temu przeciwdziałać, postanowiliśmy zastosować ważenie poszczególnych klas. Zbadaliśmy też, jaki wpływ na wyniki będzie miała augmentacja danych - w tym celu wykorzystaliśmy bibliotekę `nlpaug`. W ramach eksperymentów porównaliśmy działanie poszczególnych architektur oraz zbadaliśmy wpływ augmentacji na wynik.

### Wyniki eksperymentów

Ostatecznie najlepsze wyniki udało się uzyskać dla dotrenowanego na danych trenujących modelu DistilBERT - ostatecznie model na danych walidacyjnych uzyskał ok. 90% accuracy. Niestety trening tego modelu również trwał najdłużej - dotrenowanie przez zaledwie 5 epok trwało ok. 30 min. Najgorzej radził sobie z zadaniem model bag of words - osiągał on wynik na danych walidacyjnych ok. 20% accuracy. Z kolei model fasttext osiągał wyniki ok. 60% accuracy. Augmentacje danych jakie testowaliśmy to augmentacje poprzez zamianę losowych słów synonimami/antonimami lub dodanie losowych literówek. Same w sobie augmentacje nie różniły się znacząco między sobą wynikami. Z kolei nadmierne dołożenie augmentacji, takie aby wyrównać liczebnościowo wszystkie klasy, prowadziło do overfittingu przy trenowaniu modelu.

### Ostateczny model

Ostatecznie wytrenowaliśmy model DistilBERT na całych dostępnych danych. W trakcie treningu zastosowaliśmy augmentację danych poprzez zamianę losowych słów synonimami w danych trenujących. Zastosowaliśmy też ważenie poszczególnych klas. 
