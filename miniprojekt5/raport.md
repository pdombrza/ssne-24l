## Miniprojekt 5

Mateusz Kiełbus, Paweł Dombrzalski

### Temat

Tematem zadania było stworzenie rekurencyjnej sieci neuronowej, mającej klasyfikować kompozytora na podstawie sekwencji akordów.

### Struktura projektu

W pliku analysis.ipynb znajduje się wstępna analiza danych. W plikach _recursion\_test_ znajdują się testy poszczególnych modeli LSTM. W plikach _conv\_lstm_ znajdują się testy modeli LSTM z warstwami konwolucyjnymi. W plikach _gru_ i _rnn_ znajdują się poszczególnie testy sieci GRU i RNN. W pliku _final\_model_ znajduje się ostateczny model, wykorzystany do wygenerowania predykcji.

### Wykonane eksperymenty

W danych treningowych klasy są niezbalansowane - aby temu przeciwdziałać, postanowiliśmy zastosować ważenie poszczególnych klas. W ramach eksperymentów sprawdzaliśmy jak różne rozmiary sieci oraz architektury wpłyną na ostateczny wynik. Głównie skupiliśmy się na sieciach LSTM, ale zbadaliśmy też działanie sieci GRU oraz RNN. Sprawdziliśmy, czy sieci dwukierunkowe będą działać lepiej niż jednokierunkowe oraz czy dodanie warstwy konwolucyjnej (Conv1D) będzie miało wpływ na skuteczność sieci. W ramach porównania modeli, badaliśmy accuracy na zbiorze walidacyjnym - wydzielonym ze zbioru treningowego.

### Wyniki eksperymentów

Ostatecznie najlepsze wyniki udało się uzyskać dla modelu wkorzystującego warstwy LSTM. Zwykłe sieci rekurencyjne oraz sieci GRU osiągały gorsze wyniki od LSTM. Dodanie dropoutu i batch normalization poprawiło działaie modelu. Sieci dwukierunkowe również produkowały lepsze wyniki. Z kolei dodanie warstwy konwolucyjnej nie miało zbytnio wpływu na skuteczność sieci - accuracy było podobne do modeli o podobnej architekturze bez tej warstwy. Zwiększenie ilości warstw rekurencyjnych oraz ilości parametrów warstwy ukrytej również miało pozytywny wpływ na accuracy modelu.


### Ostateczny model

Do wygenerowania ostatecznych predykcji zdecydowaliśmy wykorzystać sieć z pliku _recursion\_test16_. Sieć zawiera 2 warstwy LSTM z 75 parametrami warstwy ukrytej. Zastosowaliśmy dropout w warstwach lstm i batch normalization po nich. Model ma też warstwę liniową. Ostatecznie sieć uzyskiwała wynik ok. 78% accuracy na zbiorze walidacyjnym.
