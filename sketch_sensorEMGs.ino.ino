float tensao;
int cont;

void setup() {
    Serial.begin(19200);  // Inicia a comunicação serial
    tensao = 0;
    cont = 0;
}

void loop() {
    cont++;
    int valorEMG = analogRead(A0);  // Lê o valor do sensor EMG
    tensao =  ((valorEMG / 1023.0) * 5.0);
    Serial.print(tensao);
    Serial.print(" ");
    delay(1);  // Pequeno atraso para evitar sobrecarga
}
