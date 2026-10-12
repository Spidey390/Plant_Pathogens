#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include "secrets.h"
const char *ssid = WIFI_SSID;
const char *password = WIFI_PASS;
const char *serverUrl =
    "https://plant-pathogens-detection.onrender.com/predict";
const char *deviceId =
    "ESP32-CAM-01";
const unsigned long interval =
    3000;
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
const int PUMP_PINS[3] = {13, 14, 15};
#define RELAY_ON LOW
#define RELAY_OFF HIGH
const unsigned long SPRAY_MS = 5000;
String lastResponse = "";
void initRelays()
{
  for (int i = 0; i < 3; i++)
  {
    digitalWrite(PUMP_PINS[i], RELAY_OFF);
    pinMode(PUMP_PINS[i], OUTPUT);
    digitalWrite(PUMP_PINS[i], RELAY_OFF);
  }
}
void allOff()
{
  for (int i = 0; i < 3; i++)
    digitalWrite(PUMP_PINS[i], RELAY_OFF);
}
int pumpForDisease(const String &d)
{
  if (d == "Blight")
    return 1;
  if (d == "Common_Rust")
    return 2;
  if (d == "Gray_Leaf_Spot")
    return 3;
  return 0;
}
String jsonStr(const String &body, const char *key)
{
  int k = body.indexOf(String("\"") + key + "\":");
  if (k < 0)
    return "";
  k = body.indexOf(':', k) + 1;
  while (k < (int)body.length() && body[k] == ' ')
    k++;
  if (k >= (int)body.length() || body[k] != '"')
    return "";
  int e = body.indexOf('"', k + 1);
  if (e < 0)
    return "";
  return body.substring(k + 1, e);
}
float jsonFloat(const String &body, const char *key)
{
  int k = body.indexOf(String("\"") + key + "\":");
  if (k < 0)
    return -1;
  k = body.indexOf(':', k) + 1;
  while (k < (int)body.length() && body[k] == ' ')
    k++;
  return body.substring(k).toFloat();
}
void sprayPump(int pump, unsigned long ms)
{
  if (pump < 1 || pump > 3)
    return;
  allOff();
  Serial.printf("SPRAY pump %d for %lu ms\n", pump, ms);
  digitalWrite(PUMP_PINS[pump - 1], RELAY_ON);
  delay(ms);
  digitalWrite(PUMP_PINS[pump - 1], RELAY_OFF);
}
void handleResult(const String &body)
{
  int pump = pumpForDisease(jsonStr(body, "disease"));
  if (pump == 0)
    return;
  sprayPump(pump, SPRAY_MS);
}
bool initCamera()
{
  camera_config_t config;
  memset(&config, 0, sizeof(config));
  config.ledc_channel =
      LEDC_CHANNEL_0;
  config.ledc_timer =
      LEDC_TIMER_0;
  config.pin_d0 =
      Y2_GPIO_NUM;
  config.pin_d1 =
      Y3_GPIO_NUM;
  config.pin_d2 =
      Y4_GPIO_NUM;
  config.pin_d3 =
      Y5_GPIO_NUM;
  config.pin_d4 =
      Y6_GPIO_NUM;
  config.pin_d5 =
      Y7_GPIO_NUM;
  config.pin_d6 =
      Y8_GPIO_NUM;
  config.pin_d7 =
      Y9_GPIO_NUM;
  config.pin_xclk =
      XCLK_GPIO_NUM;
  config.pin_pclk =
      PCLK_GPIO_NUM;
  config.pin_vsync =
      VSYNC_GPIO_NUM;
  config.pin_href =
      HREF_GPIO_NUM;
  config.pin_sscb_sda =
      SIOD_GPIO_NUM;
  config.pin_sscb_scl =
      SIOC_GPIO_NUM;
  config.pin_pwdn =
      PWDN_GPIO_NUM;
  config.pin_reset =
      RESET_GPIO_NUM;
  config.xclk_freq_hz =
      8000000;
  config.pixel_format =
      PIXFORMAT_JPEG;
  config.grab_mode =
      CAMERA_GRAB_LATEST;
  if (psramFound())
  {
    Serial.println(
        "PSRAM detected");
    config.frame_size =
        FRAMESIZE_QVGA;
    config.jpeg_quality =
        15;
    config.fb_count =
        1;
    config.fb_location =
        CAMERA_FB_IN_PSRAM;
  }
  else
  {
    Serial.println(
        "PSRAM NOT detected");
    config.frame_size =
        FRAMESIZE_QQVGA;
    config.jpeg_quality =
        18;
    config.fb_count =
        1;
    config.fb_location =
        CAMERA_FB_IN_DRAM;
  }
  Serial.println(
      "Initializing camera...");
  esp_err_t err =
      esp_camera_init(&config);
  if (err != ESP_OK)
  {
    Serial.printf(
        "Camera init failed: 0x%x\n",
        err);
    return false;
  }
  Serial.println(
      "Camera initialized successfully");
  sensor_t *sensor =
      esp_camera_sensor_get();
  if (sensor != nullptr)
  {
    sensor->set_framesize(
        sensor,
        FRAMESIZE_QVGA);
    sensor->set_quality(
        sensor,
        15);
  }
  Serial.println(
      "Camera OK");
  return true;
}
bool scanForWiFi()
{
  Serial.println();
  Serial.println(
      "Scanning WiFi...");
  int found =
      WiFi.scanNetworks();
  if (found <= 0)
  {
    Serial.println(
        "No WiFi networks found");
    return false;
  }
  bool seen = false;
  for (int i = 0; i < found; i++)
  {
    Serial.printf(
        "[%d] %s | CH %d | RSSI %d dBm\n",
        i,
        WiFi.SSID(i).c_str(),
        WiFi.channel(i),
        WiFi.RSSI(i));
    if (
        WiFi.SSID(i) ==
        String(ssid))
    {
      seen = true;
    }
  }
  WiFi.scanDelete();
  if (seen)
  {
    Serial.println(
        "Target SSID found");
  }
  else
  {
    Serial.println(
        "Target SSID NOT found");
  }
  return seen;
}
bool connectToWiFi()
{
  WiFi.mode(
      WIFI_STA);
  WiFi.setAutoReconnect(
      false);
  WiFi.persistent(
      false);
  if (!scanForWiFi())
  {
    return false;
  }
  for (
      int attempt = 1;
      attempt <= 5;
      attempt++)
  {
    Serial.println();
    Serial.printf(
        "Connecting to WiFi "
        "attempt %d/5\n",
        attempt);
    WiFi.disconnect(
        false,
        false);
    delay(500);
    WiFi.begin(
        ssid,
        password);
    unsigned long start =
        millis();
    while (
        WiFi.status() != WL_CONNECTED &&
        millis() - start < 20000)
    {
      delay(500);
      Serial.print(".");
    }
    Serial.println();
    if (
        WiFi.status() ==
        WL_CONNECTED)
    {
      Serial.println();
      Serial.println(
          "==============================");
      Serial.println(
          "WiFi CONNECTED");
      Serial.print(
          "IP: ");
      Serial.println(
          WiFi.localIP());
      Serial.print(
          "RSSI: ");
      Serial.print(
          WiFi.RSSI());
      Serial.println(
          " dBm");
      Serial.println(
          "==============================");
      return true;
    }
    Serial.printf(
        "WiFi failed. Status = %d\n",
        WiFi.status());
    WiFi.disconnect(
        false,
        false);
    delay(2000);
  }
  return false;
}
camera_fb_t *captureImage()
{
  Serial.println();
  Serial.println(
      "Capturing image...");
  delay(200);
  camera_fb_t *fb =
      esp_camera_fb_get();
  if (!fb)
  {
    Serial.println(
        "Camera capture failed");
    return nullptr;
  }
  Serial.printf(
      "Image captured: %u bytes\n",
      fb->len);
  Serial.printf(
      "Image width: %d\n",
      fb->width);
  Serial.printf(
      "Image height: %d\n",
      fb->height);
  return fb;
}
void sendImage(
    camera_fb_t *fb)
{
  if (!fb)
  {
    return;
  }
  if (
      WiFi.status() !=
      WL_CONNECTED)
  {
    Serial.println(
        "WiFi not connected");
    return;
  }
  WiFiClientSecure client;
  client.setInsecure();
  client.setTimeout(
      60);
  HTTPClient http;
  Serial.println(
      "Connecting to Render server...");
  if (
      !http.begin(
          client,
          serverUrl))
  {
    Serial.println(
        "HTTP begin failed");
    return;
  }
  http.setTimeout(
      60000);
  String boundary =
      "ESP32CAMBoundary";
  String filePart =
      "--" + boundary + "\r\n"
                        "Content-Disposition: form-data; "
                        "name=\"file\"; "
                        "filename=\"photo.jpg\"\r\n"
                        "Content-Type: image/jpeg\r\n\r\n";
  String devicePart =
      "\r\n--" + boundary + "\r\n"
                            "Content-Disposition: form-data; "
                            "name=\"device_id\"\r\n\r\n" +
      String(deviceId) + "\r\n--" + boundary + "\r\n"
                                               "Content-Disposition: form-data; "
                                               "name=\"api_key\"\r\n\r\n" +
      String(API_KEY) + "\r\n--" + boundary + "--\r\n";
  size_t totalLen =
      filePart.length() + fb->len + devicePart.length();
  Serial.printf(
      "Upload size: %u bytes\n",
      (unsigned int)totalLen);
  http.addHeader(
      "Content-Type",
      "multipart/form-data; boundary=" +
          boundary);
  http.addHeader(
      "Content-Length",
      String(totalLen));
  uint8_t *payload = nullptr;
  if (psramFound())
  {
    payload =
        (uint8_t *)ps_malloc(
            totalLen);
  }
  else
  {
    payload =
        (uint8_t *)malloc(
            totalLen);
  }
  if (!payload)
  {
    Serial.println(
        "Payload memory allocation failed");
    http.end();
    return;
  }
  memcpy(
      payload,
      filePart.c_str(),
      filePart.length());
  memcpy(
      payload +
          filePart.length(),
      fb->buf,
      fb->len);
  memcpy(
      payload +
          filePart.length() +
          fb->len,
      devicePart.c_str(),
      devicePart.length());
  Serial.println();
  Serial.println(
      "Sending image to server...");
  int httpCode =
      http.POST(
          payload,
          totalLen);
  free(
      payload);
  if (httpCode > 0)
  {
    Serial.printf(
        "HTTP Response: %d\n",
        httpCode);
    String response =
        http.getString();
    lastResponse = response;
    Serial.println(
        "Server Response:");
    Serial.println(
        response);
  }
  else
  {
    Serial.println(
        "POST failed");
    Serial.println(
        http.errorToString(
            httpCode));
  }
  http.end();
}
void setup()
{
  Serial.begin(
      115200);
  delay(
      1000);
  Serial.println();
  Serial.println(
      "==============================");
  Serial.println(
      "ESP32-CAM PLANT PATHOGENS");
  Serial.println(
      "==============================");
  initRelays();
  Serial.printf(
      "Free heap: %u bytes\n",
      ESP.getFreeHeap());
  Serial.printf(
      "PSRAM: %s\n",
      psramFound()
          ? "YES"
          : "NO");
  if (psramFound())
  {
    Serial.printf(
        "Free PSRAM: %u bytes\n",
        ESP.getFreePsram());
  }
  Serial.println();
  Serial.println(
      "Connecting to WiFi...");
  if (!connectToWiFi())
  {
    Serial.println(
        "WiFi connection failed");
    delay(
        5000);
    ESP.restart();
  }
  delay(
      500);
  if (!initCamera())
  {
    Serial.println(
        "Camera initialization failed");
    delay(
        5000);
    ESP.restart();
  }
  delay(
      1000);
  Serial.println();
  Serial.println(
      "Waking up Render server...");
  WiFiClientSecure warmup;
  warmup.setInsecure();
  warmup.setTimeout(
      60);
  HTTPClient h;
  if (
      h.begin(
          warmup,
          "https://plant-pathogens-detection.onrender.com/results"))
  {
    h.setTimeout(
        60000);
    int code =
        h.GET();
    Serial.printf(
        "Render response: %d\n",
        code);
    h.end();
  }
  else
  {
    Serial.println(
        "Render warmup failed");
  }
  Serial.println();
  Serial.println(
      "Server warmed up");
  Serial.println(
      "System ready");
  Serial.println();
}
void loop()
{
  if (
      WiFi.status() !=
      WL_CONNECTED)
  {
    Serial.println(
        "WiFi disconnected");
    if (!connectToWiFi())
    {
      delay(
          5000);
      return;
    }
  }
  camera_fb_t *fb =
      captureImage();
  if (!fb)
  {
    Serial.println(
        "Capture failed");
    delay(
        3000);
    return;
  }
  lastResponse = "";
  sendImage(
      fb);
  esp_camera_fb_return(
      fb);
  if (lastResponse.length() > 0)
    handleResult(lastResponse);
  Serial.println();
  Serial.printf(
      "Waiting %lu seconds...\n",
      interval / 1000);
  delay(
      interval);
}