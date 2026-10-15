#pragma once

#include "chat.h"
#include "common.h"
#include "mtmd.h"

#include <functional>
#include <string>

namespace liquid {
namespace audio {

using generated_audio_t = std::vector<int16_t>;
using text_callback_t   = std::function<void(const std::string &)>;
using audio_callback_t  = std::function<void(const std::vector<int16_t> &)>;

class Runner {
  public:
    // handling depends on system prompt
    static constexpr const char *                asr_system_prompt         = "Perform ASR.";
    static constexpr const char *                interleaved_system_prompt = "Respond with interleaved text and audio.";
    static inline const std::vector<std::string> tts_system_prompts        = {
        "Perform TTS. Use the US male voice.",
        "Perform TTS. Use the UK male voice.",
        "Perform TTS. Use the US female voice.",
        "Perform TTS. Use the UK female voice.",
    };

    struct Message {
        std::string            role;
        std::string            content;
        std::vector<std::byte> wav;
        // assistant and tool turns, as the OpenAI API carries them; rendered by the chat template
        std::vector<common_chat_tool_call> tool_calls;
        std::string                        tool_call_id;
        std::string                        tool_name;
    };

    Runner();
    ~Runner();

    void reset();

    int  init(common_params params);
    void stop();
    int  generate(const std::vector<Message> &              messages,
                  int                                       n_predict,
                  const text_callback_t &                   text_callback,
                  const audio_callback_t &                  audio_callback,
                  const std::vector<mtmd_output_modality> & modalities,
                  const std::vector<common_chat_tool> &     tools          = {},
                  bool                                      special_tokens = false);

    int          get_output_sample_rate() const;
    const char * get_last_error() const;
  private:
    class RunnerImpl;
    std::unique_ptr<RunnerImpl> impl_;
};

}  // namespace audio
}  // namespace liquid
