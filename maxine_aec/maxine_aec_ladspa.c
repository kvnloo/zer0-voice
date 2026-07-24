#include <ladspa.h>
#include <nvAudioEffects.h>
#include <aec.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MAXINE_AEC_MODEL
#error "MAXINE_AEC_MODEL must name the TensorRT AEC model"
#endif

enum {
  PORT_NEAR = 0,
  PORT_FAR,
  PORT_OUTPUT,
  PORT_INTENSITY,
  PORT_COUNT
};

enum {
  FRAME = 480,
  QUEUE_CAPACITY = FRAME * 4
};

typedef struct {
  const LADSPA_Data *near_end;
  const LADSPA_Data *far_end;
  LADSPA_Data *output;
  const LADSPA_Data *intensity;
  NvAFX_Handle effect;
  float near_frame[FRAME];
  float far_frame[FRAME];
  float effect_frame[FRAME];
  float queue[QUEUE_CAPACITY];
  unsigned frame_used;
  unsigned queue_read;
  unsigned queue_write;
  unsigned queue_count;
  float applied_intensity;
  int ready;
} MaxineAec;

static void queue_push(MaxineAec *self, float value) {
  if (self->queue_count == QUEUE_CAPACITY) {
    self->queue_read = (self->queue_read + 1U) % QUEUE_CAPACITY;
    self->queue_count--;
  }
  self->queue[self->queue_write] = value;
  self->queue_write = (self->queue_write + 1U) % QUEUE_CAPACITY;
  self->queue_count++;
}

static float queue_pop(MaxineAec *self) {
  float value;
  if (self->queue_count == 0U) {
    return 0.0F;
  }
  value = self->queue[self->queue_read];
  self->queue_read = (self->queue_read + 1U) % QUEUE_CAPACITY;
  self->queue_count--;
  return value;
}

static int require_status(const char *operation, NvAFX_Status status) {
  if (status == NVAFX_STATUS_SUCCESS) {
    return 1;
  }
  fprintf(stderr, "zer0-maxine-aec: %s failed (%d)\n", operation, status);
  return 0;
}

static void sdk_log(LoggingSeverity level, const char *message, void *userdata) {
  (void)userdata;
  fprintf(stderr, "zer0-maxine-aec: NVIDIA %s: %s\n",
          LogSeverityToString(level), message);
}

static int initialize_effect(MaxineAec *self) {
  const char *models[] = {MAXINE_AEC_MODEL};
  NvAFX_Status status;

  (void)NvAFX_InitializeLogger(LOG_LEVEL_INFO, LOG_TARGET_CALLBACK, "", sdk_log, NULL);
  status = NvAFX_CreateEffect(NVAFX_EFFECT_AEC, &self->effect);
  if (status != NVAFX_STATUS_SUCCESS) {
    fprintf(stderr, "zer0-maxine-aec: NvAFX_CreateEffect failed (%d)\n", status);
    return 0;
  }
  if (!require_status("select current CUDA device",
                      NvAFX_SetU32(self->effect, NVAFX_PARAM_USE_DEFAULT_GPU, 0U)) ||
      !require_status("set 48 kHz input",
                      NvAFX_SetU32(self->effect, NVAFX_PARAM_INPUT_SAMPLE_RATE, 48000U)) ||
      !require_status("set model path",
                      NvAFX_SetStringList(self->effect, NVAFX_PARAM_MODEL_PATH, models, 1U)) ||
      !require_status("set stream count",
                      NvAFX_SetU32(self->effect, NVAFX_PARAM_NUM_STREAMS, 1U)) ||
      !require_status("set 480-sample frame",
                      NvAFX_SetU32(self->effect, NVAFX_PARAM_NUM_SAMPLES_PER_INPUT_FRAME, FRAME)) ||
      !require_status("load effect", NvAFX_Load(self->effect))) {
    NvAFX_DestroyEffect(self->effect);
    self->effect = NULL;
    return 0;
  }
  self->applied_intensity = 1.0F;
  (void)NvAFX_SetFloat(self->effect, NVAFX_PARAM_INTENSITY_RATIO, self->applied_intensity);
  return 1;
}

static LADSPA_Handle instantiate(const LADSPA_Descriptor *descriptor, unsigned long sample_rate) {
  MaxineAec *self;
  unsigned i;
  (void)descriptor;
  if (sample_rate != 48000UL) {
    fprintf(stderr, "zer0-maxine-aec: 48 kHz required, got %lu\n", sample_rate);
    return NULL;
  }
  self = calloc(1U, sizeof(*self));
  if (self == NULL) {
    return NULL;
  }
  self->ready = initialize_effect(self);
  if (!self->ready) {
    free(self);
    return NULL;
  }
  /* One model frame of bounded startup latency prevents output underruns. */
  for (i = 0U; i < FRAME; i++) {
    queue_push(self, 0.0F);
  }
  return self;
}

static void connect_port(LADSPA_Handle instance, unsigned long port, LADSPA_Data *data) {
  MaxineAec *self = instance;
  switch (port) {
  case PORT_NEAR: self->near_end = data; break;
  case PORT_FAR: self->far_end = data; break;
  case PORT_OUTPUT: self->output = data; break;
  case PORT_INTENSITY: self->intensity = data; break;
  default: break;
  }
}

static void activate(LADSPA_Handle instance) {
  MaxineAec *self = instance;
  unsigned i;
  self->frame_used = 0U;
  self->queue_read = 0U;
  self->queue_write = 0U;
  self->queue_count = 0U;
  for (i = 0U; i < FRAME; i++) {
    queue_push(self, 0.0F);
  }
  if (self->effect != NULL) {
    (void)NvAFX_Reset(self->effect, NULL, 1U);
  }
}

static void run(LADSPA_Handle instance, unsigned long sample_count) {
  MaxineAec *self = instance;
  unsigned long i;
  float requested;

  if (!self->ready || self->near_end == NULL || self->far_end == NULL || self->output == NULL) {
    if (self->output != NULL) {
      memset(self->output, 0, sample_count * sizeof(*self->output));
    }
    return;
  }

  requested = self->intensity == NULL ? 1.0F : fmaxf(0.0F, fminf(1.0F, *self->intensity / 100.0F));
  if (fabsf(requested - self->applied_intensity) > 0.001F) {
    if (NvAFX_SetFloat(self->effect, NVAFX_PARAM_INTENSITY_RATIO, requested) == NVAFX_STATUS_SUCCESS) {
      self->applied_intensity = requested;
    }
  }

  for (i = 0UL; i < sample_count; i++) {
    const float *inputs[2];
    float *outputs[1];
    unsigned j;

    self->near_frame[self->frame_used] = self->near_end[i];
    self->far_frame[self->frame_used] = self->far_end[i];
    self->frame_used++;

    if (self->frame_used == FRAME) {
      inputs[0] = self->near_frame;
      inputs[1] = self->far_frame;
      outputs[0] = self->effect_frame;
      if (NvAFX_Run(self->effect, inputs, outputs, FRAME, 2U) == NVAFX_STATUS_SUCCESS) {
        for (j = 0U; j < FRAME; j++) {
          queue_push(self, self->effect_frame[j]);
        }
      } else {
        fprintf(stderr, "zer0-maxine-aec: NvAFX_Run failed\n");
        for (j = 0U; j < FRAME; j++) {
          queue_push(self, 0.0F);
        }
      }
      self->frame_used = 0U;
    }
    self->output[i] = queue_pop(self);
  }
}

static void cleanup(LADSPA_Handle instance) {
  MaxineAec *self = instance;
  if (self != NULL) {
    if (self->effect != NULL) {
      NvAFX_DestroyEffect(self->effect);
    }
    free(self);
  }
}

static LADSPA_PortDescriptor port_descriptors[PORT_COUNT] = {
  LADSPA_PORT_INPUT | LADSPA_PORT_AUDIO,
  LADSPA_PORT_INPUT | LADSPA_PORT_AUDIO,
  LADSPA_PORT_OUTPUT | LADSPA_PORT_AUDIO,
  LADSPA_PORT_INPUT | LADSPA_PORT_CONTROL
};
static const char *port_names[PORT_COUNT] = {
  "Near End", "Far End", "Output", "Intensity"
};
static LADSPA_PortRangeHint port_hints[PORT_COUNT] = {
  {0, 0.0F, 0.0F},
  {0, 0.0F, 0.0F},
  {0, 0.0F, 0.0F},
  {LADSPA_HINT_BOUNDED_BELOW | LADSPA_HINT_BOUNDED_ABOVE | LADSPA_HINT_DEFAULT_MAXIMUM,
   0.0F, 100.0F}
};
static LADSPA_Descriptor descriptor = {
  16682995UL,
  "zer0-maxine-aec",
  LADSPA_PROPERTY_HARD_RT_CAPABLE,
  "Zer0 NVIDIA Maxine AEC",
  "Zer0",
  "Proprietary NVIDIA SDK runtime; plugin glue MIT",
  PORT_COUNT,
  port_descriptors,
  port_names,
  port_hints,
  NULL,
  instantiate,
  connect_port,
  activate,
  run,
  NULL,
  NULL,
  NULL,
  cleanup
};

const LADSPA_Descriptor *ladspa_descriptor(unsigned long index) {
  return index == 0UL ? &descriptor : NULL;
}
