// EEG Acquisition for BioAmp EXG Pill
// Sampling: 250 Hz

const int eegPin = A0;
unsigned long lastMicros = 0;
const unsigned long interval = 1000000UL / 250;  // 250 Hz

void setup() {
  Serial.begin(115200);
}

void loop() {
  if (micros() - lastMicros >= interval) {
    lastMicros += interval;

    int raw = analogRead(eegPin);
    Serial.println(raw);
  }
}
