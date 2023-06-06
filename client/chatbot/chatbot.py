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

def format_initial_prompt(scenario):
    lines = []

    for record in list(scenario.example) + [scenario.setup]:
        if lines:
            lines.append("")
        lines.append(f"=== NEW CHAT ===")
        if record.context.description:
            lines.append("")
            lines.append(record.context.description)
        lines.append("")
        for line in record.line:
            lines.append(f"{line.speaker}: {line.text}")
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

def is_acceptable_token(token, exclude=set(), require_prefix=""):
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
    
    return True

def choose_softmax(logits, temperature=0.5, exclude=set(), require_prefix=""):
    choices = [(math.exp(logit.logit / temperature), logit.token) for logit in logits if is_acceptable_token(logit.token, exclude=exclude, require_prefix="")]
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
        logging.info(f"Feeding: {repr(input_str)}")
        response = stub.DoAddTokensAndCompute(llama_pb2.DoAddTokensAndComputeRequest(
            input_tokens = llama_pb2.InputTokens(
                str = input_str,
            ),
            top_n_logits = 40,
        ))
        if result.endswith("\n"):
            break
        exclude = {13} if not result else set()
        chosen_token = choose_softmax(response.logit, temperature=0.5, exclude=exclude, require_prefix=require_prefix)
        logging.info(f"Chose token: {chosen_token}")
        input_str = chosen_token.token_str.decode("utf-8")
        require_prefix = require_prefix[len(input_str):]
        result += input_str

    assert result.startswith(original_require_prefix)
    result = result[len(original_require_prefix):]
    
    return result.strip(), response.context_size_tokens

def main(argv):
  del argv  # Unused.

  with open(FLAGS.scenario) as f:
    scenario = text_format.Parse(f.read(), chatbot_pb2.Scenario())
  
  logging.info(f"Connecting to: {FLAGS.server}")
  with grpc.insecure_channel(FLAGS.server) as channel:
    stub = llama_pb2_grpc.LlamaServiceStub(channel)

    stub.DoLoadModel(llama_pb2.DoLoadModelRequest(model_name=FLAGS.model_name))

    prompt = format_initial_prompt(scenario)
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
        output_line, current_context_size = generate_line(stub, human_speaker, input_line, bot_speaker)
        print(f"[{current_context_size}] {bot_speaker}: {output_line}")
        sys.stdout.flush()

if __name__ == '__main__':
  app.run(main)