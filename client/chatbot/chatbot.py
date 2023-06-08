from absl import app
from absl import flags
from absl import logging

from google.protobuf import text_format

from proto import llama_pb2_grpc
from proto import llama_pb2
from client.chatbot.proto import chatbot_pb2

import grpc
import math
import sys
import random

FLAGS = flags.FLAGS
flags.DEFINE_string("scenario", "scenario.pb_text", "Scenario to run.")
flags.DEFINE_string("server", "localhost:50051", "Address of server to connect to.")
flags.DEFINE_string("model_name", "13B", "Model to request.")

def max_prompt_size(scenario):
    n = 0
    for example in scenario.example:
        if example.sticky:
            continue
        n += len(example.record.line)
    n += len(scenario.setup.line)
    return n

def format_initial_prompt(scenario, size=None):
    linesets = []

    size = size if size is not None else max_prompt_size(scenario)

    def show_record(record, end_chat_marker, show_only=None):
        if (show_only is not None) and (show_only <= 0):
            return
        lines = []
        linesets.append(lines)
        lines.append(f"=== NEW CHAT ===")
        if record.context.description:
            lines.append("")
            lines.append(record.context.description)
        lines.append("")
        omit = max(0, len(record.line) - show_only if show_only is not None else 0)
        if omit:
            lines.append("...")
        for i, line in enumerate(record.line):
            if i < omit:
                continue
            lines.append(f"{line.speaker}: {line.text}")
        if end_chat_marker:
            lines.append(f"<<< {record.context.human.nickname} ended chat >>>")
    
    remaining = size

    show_record(scenario.setup, end_chat_marker=False, show_only=remaining)
    remaining -= len(scenario.setup.line)

    for example in reversed(scenario.example):
        show_record(example.record, end_chat_marker=example.has_end_chat_marker, show_only=remaining)
        remaining -= len(example.record.line)

    linesets.reverse()
    lines = []
    for lineset in linesets:
        if lines:
            lines.append("")
        lines.extend(lineset)
        lines.append("")
    
    return "\n".join(lines)

def read_input(human_speaker):
    sys.stdout.write(f"[{human_speaker}] >>> ")
    sys.stdout.flush()
    rv = sys.stdin.readline().strip()
    if not rv:
        sys.stdout.write(f"::: Goodbye!")
        sys.stdout.flush()
        sys.exit(0)
    return rv

def is_acceptable_token(token, exclude=set(), require_prefix="", forbid_prefixes=set()):
    if token.token_id in exclude:
        return False
    
    try:
        value = token.token_str.decode("utf-8")
    except UnicodeDecodeError:
        return False
    
    if require_prefix:
        n = min(len(require_prefix), len(value))
        if value[:n] != require_prefix[:n]:
            return False
    
    for prefix in forbid_prefixes:
        if value.startswith(prefix):
            return False
    
    return True

def choose_softmax(logits, temperature=0.5, exclude=set(), require_prefix="", forbid_prefixes=set()):
    choices = [(math.exp(logit.logit / temperature), logit.token) for logit in logits if is_acceptable_token(logit.token, exclude=exclude, require_prefix=require_prefix, forbid_prefixes=forbid_prefixes)]
    total = sum([choice[0] for choice in choices])
    x = random.random() * total
    for choice in choices:
        x -= choice[0]
        if x <= 0:
            break
    return choice[1]


def generate_line(stub, human_speaker, input_line, bot_speaker):
    # Note: there is a subtlety here. Why not just f"{human_speaker}: {input_line}\n{bot_speaker}: "?
    #       The reason is that there is no guarantee that the following will hold:
    #            Tokenize(s1 + s2) = Tokenize(s1) + Tokenize(s2)        [NOT true in general]
    #       By injecting the f"{bot_speaker}: " prefix, we may be making the natural tokenization
    #       of the natural line to generate impossible. (For instance: " My" is a token, but
    #       since we included the space already, we're forcing the model to generate either two spaces
    #       in a row. This is a key example -- many tokens start with spaces.)
    #       By instead letting the model generate its own line, but fully constraining the start of it,
    #       we will always get the proper tokenization.
    #       (Note: we are still assuming that a newline is always a token by itself. I believe this
    #       is at least currently true.)

    input_str = f"{human_speaker}: {input_line}\n"
    original_require_prefix = f"{bot_speaker}: "
    require_prefix = original_require_prefix

    result = ""

    while True:
        logging.debug(f"Feeding: {repr(input_str)}")
        response = stub.DoAddTokensAndCompute(llama_pb2.DoAddTokensAndComputeRequest(
            input_tokens = llama_pb2.InputTokens(
                str = input_str,
            ),
            top_n_logits = 40,
        ))
        if result.endswith("\n"):
            break
        exclude = {13} if not result else set()
        forbid_prefixes = {" ", "<"} if not result else set()
        chosen_token = choose_softmax(response.logit, temperature=0.5, exclude=exclude, require_prefix=require_prefix, forbid_prefixes=forbid_prefixes)
        logging.debug(f"Chose token: {chosen_token}")
        input_str = chosen_token.token_str.decode("utf-8")
        require_prefix = require_prefix[len(input_str):]
        result += input_str

    assert result.startswith(original_require_prefix)
    result = result[len(original_require_prefix):]
    
    return result.strip(), response.context_size_tokens

def format_suitable_prompt(scenario, target_context_size, count_tokens):
    min_size = 2
    max_size = max_prompt_size(scenario)

    assert count_tokens(format_initial_prompt(scenario, size=min_size)) <= target_context_size

    lo = min_size
    hi = max_size + 1

    assert hi >= lo

    while hi > lo + 1:
        mid = (lo + hi) // 2
        ntok = count_tokens(format_initial_prompt(scenario, size=mid))
        if ntok <= target_context_size:
            logging.debug("Tried size=%d; too small (%d tokens)", mid, ntok)
            lo = mid
        else:
            logging.debug("Tried size=%d; too big (%d tokens)", mid, ntok)
            hi = mid

    prompt = format_initial_prompt(scenario, size=lo)

    logging.debug("Found suitable prompt size: %d [between %d and %d]; %d tokens", lo, min_size, max_size, count_tokens(prompt))

    return prompt

def main(argv):
  del argv  # Unused.

  with open(FLAGS.scenario) as f:
    scenario = text_format.Parse(f.read(), chatbot_pb2.Scenario())
  
  logging.info(f"Connecting to: {FLAGS.server}")
  with grpc.insecure_channel(FLAGS.server) as channel:
    stub = llama_pb2_grpc.LlamaServiceStub(channel)

    stub.DoLoadModel(llama_pb2.DoLoadModelRequest(model_name=FLAGS.model_name))

    def count_tokens(s):
        response = stub.Tokenize(llama_pb2.TokenizeRequest(text=s))
        return len(response.token)

    min_size = 2
    max_size = max_prompt_size(scenario)

    max_context_size = 2048
    target_context_size = max_context_size // 2
    context_size_threshold = max_context_size - 256

    target_context_size = 400
    context_size_threshold = 512

    prompt = format_suitable_prompt(scenario, target_context_size, count_tokens)

    human_speaker = scenario.setup.context.human.nickname
    bot_speaker = scenario.setup.context.bot.nickname
    opposite_speaker = {
        human_speaker: bot_speaker,
        bot_speaker: human_speaker,
    }

    logging.info(f"Feeding: {repr(prompt)}")
    stub.DoAddTokensAndCompute(llama_pb2.DoAddTokensAndComputeRequest(
        input_tokens = llama_pb2.InputTokens(
            str = prompt,
        ),
    ))

    for line in scenario.setup.line:
        print(f"{line.speaker}: {line.text}")
    sys.stdout.flush()

    while True:
        input_line = read_input(human_speaker)
        scenario.setup.line.add(speaker=human_speaker, text=input_line)
        output_line, current_context_size = generate_line(stub, human_speaker, input_line, bot_speaker)
        scenario.setup.line.add(speaker=bot_speaker, text=output_line)
        print(f"[{current_context_size}] {bot_speaker}: {output_line}")
        sys.stdout.flush()

        if current_context_size > context_size_threshold:
            logging.info("Threshold reached; recomputing prompt.")
            prompt = format_suitable_prompt(scenario, target_context_size, count_tokens)
            stub.DoAddTokensAndCompute(llama_pb2.DoAddTokensAndComputeRequest(
                input_tokens = llama_pb2.InputTokens(
                    str = prompt,
                ),
                clear_context_first = True,
            ))

if __name__ == '__main__':
  app.run(main)
