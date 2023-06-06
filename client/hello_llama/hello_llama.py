from proto import llama_pb2_grpc
from proto import llama_pb2
import grpc
import math
import sys
import random

SAMPLE = """
Host: Hello and thanks for listening! Let's get right to it. Today's guests, could you introduce yourselves?
Cleopatra: Hi there, I'm Cleopatra VII Philopator, Queen of Egypt, Pharaoh of the Ptolemaic Kingdom, and the last active ruler of the Ptolemaic dynasty.
Einstein: I'm Albert Einstein, a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics.
Genghis: I'm Genghis Khan, founder and first Great Khan of the Mongol Empire.
GRRM: I'm George R. R. Martin, an American novelist and short story writer in the fantasy, horror, and science fiction genres, screenwriter, and television producer.
Host: Wonderful. Today our topic is microservices vs. monoliths.
""".strip()

def choose_softmax(logits, temperature=0.5):
    print(logits)
    choices = [(math.exp(logit.logit / temperature), logit.token) for logit in logits]
    total = sum([choice[0] for choice in choices])
    x = random.random() * total
    for choice in choices:
        x -= choice[0]
        if x <= 0:
            break
    return choice[1]

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = llama_pb2_grpc.LlamaServiceStub(channel)

        stub.DoLoadModel(llama_pb2.DoLoadModelRequest(
            model_name="13B",
        ))

        response = stub.Tokenize(llama_pb2.TokenizeRequest(
            text = SAMPLE
        ))
        for token in response.token:
            print(token.token_id, repr(token.token_str))

        input_str = SAMPLE
        sys.stdout.write(SAMPLE)
        sys.stdout.write("[END PROMPT]")

        newline_counter = 0 
        while True:
            response = stub.DoAddTokensAndCompute(llama_pb2.DoAddTokensAndComputeRequest(
                input_tokens = llama_pb2.InputTokens(
                    str = input_str,
                ),
                top_n_logits = 40,
            ))
            chosen_token = choose_softmax(response.logit, temperature=0.8)
            input_str = chosen_token.token_str
            sys.stdout.write(chosen_token.token_str.decode("utf-8"))
            if chosen_token.token_id == 13:
                newline_counter += 1
                if newline_counter == 1:
                    stub.DoSaveCheckpoint(llama_pb2.DoSaveCheckpointRequest())
                    sys.stdout.write("[save checkpoint]\n\n")
                if newline_counter >= 5:
                    break
            sys.stdout.flush()
        
        while True:
            newline_counter = 0

            sys.stdout.write("\n\n[restore checkpoint]\n")
            stub.DoRestoreCheckpoint(llama_pb2.DoRestoreCheckpointRequest())

            while True:
                response = stub.DoAddTokensAndCompute(llama_pb2.DoAddTokensAndComputeRequest(
                    input_tokens = llama_pb2.InputTokens(
                        str = input_str,
                    ),
                    top_n_logits = 40,
                ))
                chosen_token = choose_softmax(response.logit, temperature=0.8)
                input_str = chosen_token.token_str
                sys.stdout.write(chosen_token.token_str.decode("utf-8"))
                if chosen_token.token_id == 13:
                    newline_counter += 1
                    if newline_counter >= 5:
                        break
                sys.stdout.flush()


if __name__ == '__main__':
    run()
