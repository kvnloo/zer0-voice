CC ?= cc
SDK := /mnt/zer0models/nvidia/maxine/maxine_linux_audio_effects_sdk_v2.1.0/Audio_Effects_SDK
AEC := $(SDK)/features/aec
CUDA := $(SDK)/external/cuda/lib
MODEL := $(AEC)/models/sm_86/aec_48k.trtpkg
BUILD := build
TARGET := $(BUILD)/maxine_aec_ladspa.so

CPPFLAGS += -I$(SDK)/nvafx/include -I$(AEC)/include -I/workspace/zer0/vendor/noisetorch-rtx/c/ladspa
CFLAGS += -O3 -fPIC -Wall -Wextra -Werror -std=c11
LDFLAGS += -shared -Wl,-rpath,$(SDK)/nvafx/lib -Wl,-rpath,$(AEC)/lib
LDLIBS += -L$(SDK)/nvafx/lib -L$(AEC)/lib -lnv_audiofx -lnv_audiofx_aec -lm

.PHONY: all clean inspect
all: $(TARGET)

$(TARGET): maxine_aec_ladspa.c
	mkdir -p $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DMAXINE_AEC_MODEL='"$(MODEL)"' $< $(LDFLAGS) $(LDLIBS) -o $@

inspect: $(TARGET)
	LD_LIBRARY_PATH="$(CUDA)" analyseplugin "$$(pwd)/$(TARGET)"

clean:
	rm -f $(TARGET)
