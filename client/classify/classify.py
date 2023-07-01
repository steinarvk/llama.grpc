from proto import llama_pb2_grpc
from proto import llama_pb2

from absl import app
from absl import flags
from absl import logging

import grpc
import sys
import random

FLAGS = flags.FLAGS
flags.DEFINE_string("server", "localhost:50051", "Address of server to connect to.")
flags.DEFINE_string("model_name", "65B/ggml-model-q4_0", "Model to use")
flags.DEFINE_string("prompt_file", "", "Prompt file (the prompt should expect classifier challenges immediately appended)")
flags.DEFINE_list("classes", [], "Classes to use")
flags.DEFINE_string("challenges_file", "", "Items to label (one on each line)")

def main(argv):
    del argv

    with grpc.insecure_channel(FLAGS.server) as channel:
        stub = llama_pb2_grpc.LlamaServiceStub(channel)

        with open(FLAGS.prompt_file, "r") as f:
            prompt = f.read()

        challenges = sys.stdin
        if FLAGS.challenges_file:
            challenges = open(FLAGS.challenges_file, "r")

        classes = list(set(FLAGS.classes))
        if len(classes) < 2:
            raise ValueError("Must specify at least two distinct classes")

        req = llama_pb2.DoPredictRequest()
        req.model_info.model_name = FLAGS.model_name
        req.logit_processing.top_n = 5

        for line in challenges:
            value = line.strip()

            remaining_options = list(classes)
            so_far = ""
            req.full_context.str = prompt + line

            while len(remaining_options) > 1:
                response = stub.DoPredict(req)
                req.session_hint.session_id = response.session_info.session_id
                req.full_context.token_ids.CopyFrom(response.full_input_context)

                allowable_tokens = []

                for logit in response.next_token_logit:
                    token = logit.token

                    try:
                        token_str = token.token_str.decode("utf-8")
                    except UnicodeDecodeError:
                        continue

                    if_chosen = so_far + token_str
                    remaining_after = [option for option in remaining_options if option.startswith(if_chosen)]

                    if remaining_after:
                        allowable_tokens.append((logit.logit, token, token_str, remaining_after))

                if not allowable_tokens:
                    logging.warning(f"No valid token for {repr(value)}")
                    break
                _, chosen_token, chosen_token_str, new_remaining_options = max(allowable_tokens)

                so_far += chosen_token_str
                remaining_options[:] = new_remaining_options

                req.full_context.token_ids.token_id.append(chosen_token.token_id)

            if len(remaining_options) == 1:
                class_name = remaining_options[0]
            else:
                class_name = "???"

            sys.stdout.write(f"{class_name}\t{value}\n")
            sys.stdout.flush()

if __name__ == '__main__':
  app.run(main)