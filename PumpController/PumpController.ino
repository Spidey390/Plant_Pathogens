// Pump controller: plain ESP32 DevKit (NOT the cam). Polls server for spray commands.
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include "secrets.h"   // WIFI_SSID, WIFI_PASS, API_KEY (gitignored)

const char* pollUrl  = "https://plant-pathogens-detection.onrender.com/pump/next";
const char* deviceId = "ESP32-CAM-01";        // camera whose sprays this controller serves
const unsigned long POLL_MS = 5000;

// Pump1=Blight(copper) Pump2=Common_Rust(sulfur) Pump3=Gray_Leaf_Spot(triazole)
const int PUMP_PINS[3] = {25, 26, 27};
#define RELAY_ON  LOW    // active-low module. Flip if yours is active-high
#define RELAY_OFF HIGH
const unsigned long MAX_SPRAY_MS = 5000;      // hard cap

void allOff() { for (int i = 0; i < 3; i++) digitalWrite(PUMP_PINS[i], RELAY_OFF); }

void sprayPump(int pump, unsigned long ms) {
  if (pump < 1 || pump > 3) return;
  if (ms > MAX_SPRAY_MS) ms = MAX_SPRAY_MS;
  allOff();
  Serial.printf("SPRAY pump %d for %lu ms\n", pump, ms);
  digitalWrite(PUMP_PINS[pump - 1], RELAY_ON);
  delay(ms);
  digitalWrite(PUMP_PINS[pump - 1], RELAY_OFF);
}

long jsonInt(const String& body, const char* key) {
  int k = body.indexOf(String("\"") + key + "\":");
  if (k < 0) return -1;
  k = body.indexOf(':', k) + 1;
  while (k < (int)body.length() && body[k] == ' ') k++;
  return body.substring(k).toInt();
}

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries++ < 30) { delay(500); Serial.print("."); }
  if (WiFi.status() != WL_CONNECTED) { Serial.println("\nWiFi fail, restart"); ESP.restart(); }
  Serial.println("\nWiFi OK " + WiFi.localIP().toString());
}

void poll() {
  if (WiFi.status() != WL_CONNECTED) { WiFi.reconnect(); return; }
  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient http;
  http.begin(client, pollUrl);
  http.setTimeout(30000);
  http.addHeader("Content-Type", "application/x-www-form-urlencoded");
  int code = http.POST(String("device_id=") + deviceId + "&api_key=" + API_KEY);
  if (code == 200) {
    String body = http.getString();
    if (body.indexOf("\"pump\"") >= 0) {
      long pump = jsonInt(body, "pump"), dur = jsonInt(body, "duration_ms");
      if (pump > 0 && dur > 0) sprayPump((int)pump, (unsigned long)dur);
    }
  } else {
    Serial.println("poll code " + String(code));
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 3; i++) { digitalWrite(PUMP_PINS[i], RELAY_OFF); pinMode(PUMP_PINS[i], OUTPUT); digitalWrite(PUMP_PINS[i], RELAY_OFF); }
  connectWiFi();
}

void loop() { poll(); delay(POLL_MS); }
