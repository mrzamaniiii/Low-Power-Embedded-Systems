#include "Arduino_BMI270_BMM150.h"
#include "model.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <math.h>

constexpr int NUM_SAMPLES = 128;
constexpr int NUM_AXES = 6;
constexpr int NUM_FEATURES = 42;
constexpr int NUM_CLASSES = 4;

const float ACC_THRESHOLD = 1.0f;

const char* CLASS_NAMES[NUM_CLASSES] = {
  "circle",
  "left_right",
  "rest",
  "up_down"
};

const float SCALER_MEAN[NUM_FEATURES] = {
  0.2035248240361862, 0.13101443281258515, 0.4087896834504097, -0.005871117590589726,
  0.41774933049554364, 3.6735372340425534, 0.004543569979306723, -0.4051204133850641,
  0.11144093590778635, 0.43320846450118783, -0.6653338321187394, -0.1487712557774354,
  3.956117021276596, 0.0027530031441219374, -0.3219971065496174, 0.07795864561612302,
  0.5537368378582153, -0.4897278831043142, -0.12387541412039006, 9.80718085106383,
  0.002278706793509008, -0.20037304768536954, 14.229078778561126, 14.569910565589337,
  -38.078793056467745, 33.49206632882991, 7.446808510638298, 36.656519621214336,
  0.9792104692376674, 11.607819968715628, 11.952036879164107, -24.2277090549469,
  28.448225378990173, 3.939494680851064, 144.59467503333346, -0.5070402856044313,
  25.529095964862947, 25.874784125926645, -43.98491775101804, 52.89605111581214,
  2.210771276595745, 251.98656402873073
};

const float SCALER_SCALE[NUM_FEATURES] = {
  0.42973531052785063, 0.13566931691171785, 0.3075007828925677, 0.5188522885003986,
  0.42332605202498247, 4.0688080552705985, 0.007145065040265584, 0.3702292880553799,
  0.07670721114791097, 0.3630782012261587, 0.502404759341139, 0.3151046360044761,
  4.3644328099151055, 0.007545665280905669, 0.5504629831551316, 0.0915509573563386,
  0.3384173079025179, 0.6177879074220489, 0.6049833940081881, 7.187081738376251,
  0.01403546117614247, 3.776181590440021, 10.60296964668935, 10.81235613455988,
  37.495932286602255, 29.19126367676174, 7.3409043119759545, 56.30049019549609,
  3.33623251787249, 21.168945094191137, 21.262734002323974, 46.61430391582741,
  61.41324592609381, 4.382406839256138, 1255.1883759328036, 4.192863682182697,
  26.889480720727338, 26.890734874802092, 49.20661012916566, 58.55208684860394,
  1.6656381367130262, 788.8121636548046
};

float sampleBuffer[NUM_SAMPLES][NUM_AXES];

tflite::AllOpsResolver resolver;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model_tflite = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int tensorArenaSize = 16 * 1024;
uint8_t tensorArena[tensorArenaSize];

float computeMean(const float* x, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i];
  return s / n;
}

float computeStd(const float* x, int n, float meanVal) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) {
    float d = x[i] - meanVal;
    s += d * d;
  }
  return sqrtf(s / n);
}

float computeRMS(const float* x, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i] * x[i];
  return sqrtf(s / n);
}

float computeMin(const float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] < m) m = x[i];
  }
  return m;
}

float computeMax(const float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > m) m = x[i];
  }
  return m;
}

void computeFrequencyFeatures(const float* x, int n, float fs, float& dominantFreq, float& peakPower) {
  const int maxBin = 16;
  peakPower = 0.0f;
  dominantFreq = 0.0f;

  for (int k = 1; k <= maxBin; k++) {
    float realPart = 0.0f;
    float imagPart = 0.0f;

    for (int t = 0; t < n; t++) {
      float angle = 2.0f * PI * k * t / n;
      realPart += x[t] * cosf(angle);
      imagPart -= x[t] * sinf(angle);
    }

    float power = realPart * realPart + imagPart * imagPart;
    if (power > peakPower) {
      peakPower = power;
      dominantFreq = (fs * k) / n;
    }
  }
}

void extractFeatures(float features[NUM_FEATURES]) {
  float axisSignal[NUM_SAMPLES];
  int idx = 0;

  for (int axis = 0; axis < NUM_AXES; axis++) {
    for (int i = 0; i < NUM_SAMPLES; i++) {
      axisSignal[i] = sampleBuffer[i][axis];
    }

    float meanVal = computeMean(axisSignal, NUM_SAMPLES);
    float stdVal = computeStd(axisSignal, NUM_SAMPLES, meanVal);
    float rmsVal = computeRMS(axisSignal, NUM_SAMPLES);
    float minVal = computeMin(axisSignal, NUM_SAMPLES);
    float maxVal = computeMax(axisSignal, NUM_SAMPLES);

    float dominantFreq, peakPower;
    computeFrequencyFeatures(axisSignal, NUM_SAMPLES, 100.0f, dominantFreq, peakPower);

    features[idx++] = meanVal;
    features[idx++] = stdVal;
    features[idx++] = rmsVal;
    features[idx++] = minVal;
    features[idx++] = maxVal;
    features[idx++] = dominantFreq;
    features[idx++] = peakPower;
  }
}

void normalizeFeatures(float features[NUM_FEATURES]) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    if (SCALER_SCALE[i] != 0.0f) {
      features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
  }
}

int argmaxOutput() {
  int best = 0;
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output->data.f[i] > output->data.f[best]) {
      best = i;
    }
  }
  return best;
}

bool waitForMotionTrigger() {
  float aX, aY, aZ;

  while (true) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      if (aSum >= ACC_THRESHOLD) {
        delay(50);
        return true;
      }
    }
  }
}

bool captureWindow() {
  float aX, aY, aZ, gX, gY, gZ;
  int samplesRead = 0;

  while (samplesRead < NUM_SAMPLES) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      sampleBuffer[samplesRead][0] = aX;
      sampleBuffer[samplesRead][1] = aY;
      sampleBuffer[samplesRead][2] = aZ;
      sampleBuffer[samplesRead][3] = gX;
      sampleBuffer[samplesRead][4] = gY;
      sampleBuffer[samplesRead][5] = gZ;

      samplesRead++;
    }
  }

  return true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("Starting final gesture classifier...");

  if (!IMU.begin()) {
    Serial.println("IMU failed");
    while (1);
  }

  model_tflite = tflite::GetModel(gesture_model_tflite);

  static tflite::MicroInterpreter static_interpreter(
      model_tflite,
      resolver,
      tensorArena,
      tensorArenaSize,
      error_reporter,
      nullptr);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model ready!");
}

void loop() {
  if (!waitForMotionTrigger()) return;
  if (!captureWindow()) return;

  float features[NUM_FEATURES];
  extractFeatures(features);
  normalizeFeatures(features);

  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(1000);
    return;
  }

  int pred = argmaxOutput();

  Serial.print("Prediction: ");
  Serial.println(CLASS_NAMES[pred]);

  Serial.print("circle: ");
  Serial.print(output->data.f[0], 6);
  Serial.print(" left_right: ");
  Serial.print(output->data.f[1], 6);
  Serial.print(" rest: ");
  Serial.print(output->data.f[2], 6);
  Serial.print(" up_down: ");
  Serial.println(output->data.f[3], 6);

  Serial.println();
  delay(1000);
}