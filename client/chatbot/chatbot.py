from absl import app
from absl import flags
from absl import logging

from google.protobuf import text_format

from proto import llama_pb2_grpc
from proto import llama_pb2
from client.chatbot.proto import chatbot_pb2

import dataclasses
import grpc
import uuid
import os
import shlex
import shutil
import math
import sys
import random
import time

FLAGS = flags.FLAGS
flags.DEFINE_string("scenario", "scenario.pb_text", "Scenario to run.")
flags.DEFINE_string("server", "localhost:50051", "Address of server to connect to.")
flags.DEFINE_string("model_name", "13B", "Model to request.")
flags.DEFINE_string("output_transcript", "", "Save transcript of conversation to this file.")
flags.DEFINE_string("checkpoint_filename", "", "Alternate filename for checkpoint file.")
flags.DEFINE_boolean("write_checkpoint", True, "Write checkpoint files adjacent to scenario file.")
flags.DEFINE_boolean("recover_checkpoint", True, "Recover from last checkpoint if present.")

def max_prompt_size(scenario):
    n = 0
    for example in scenario.example:
        if example.sticky:
            continue
        n += len(example.record.line)
    n += len(scenario.setup.line)
    return n

def parse_speakers(speakers: list[chatbot_pb2.Speaker]) -> dict[str, chatbot_pb2.Speaker]:
    rv : dict[str, chatbot_pb2.Speaker] = {}
    used_prefixes = set()

    for speaker in speakers:
        speaker_copy = chatbot_pb2.Speaker()
        speaker_copy.CopyFrom(speaker)

        if not speaker.name:
            raise RuntimeError("nameless speaker")

        if speaker.name in rv:
            raise RuntimeError(f"duplicate speaker name: {speaker.name}")

        if not speaker_copy.HasField("affixes"):
            speaker_copy.affixes.prefix = f"{speaker.name}: "
            assert speaker_copy.HasField("affixes")

        if not speaker_copy.affixes.prefix:
            raise RuntimeError(f"speaker {speaker.name} has no prefix")

        if "\n" in speaker_copy.affixes.prefix or "\n" in speaker_copy.affixes.suffix:
            raise RuntimeError(f"speaker {speaker.name} has newline in prefix or suffix")

        if speaker_copy.affixes.prefix in used_prefixes:
            raise RuntimeError(f"speaker {speaker.name} has duplicate prefix")

        used_prefixes.add(speaker_copy.affixes.prefix)

        rv[speaker.name] = speaker_copy
    
    return rv

def format_initial_prompt(scenario, size=None):
    linesets = []

    size = size if size is not None else max_prompt_size(scenario)

    def show_record(record, end_chat_marker, show_only=None):
        speakers = parse_speakers(record.context.speaker)

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
            speaker = speakers[line.speaker]
            lines.append(f"{speaker.affixes.prefix}{line.text}{speaker.affixes.suffix}")
        if end_chat_marker:
            humans = [speaker for speaker in speakers.values() if speaker.human]
            if humans:
                human = humans[0]
                lines.append(f"<<< {human.name} ended chat >>>")
    
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

@dataclasses.dataclass
class ChatInput:
    speaker: chatbot_pb2.Speaker
    text: str

@dataclasses.dataclass
class Command:
    cmd: str
    args: list[str]

def read_input(prompt, speakers):
    sys.stdout.write(f"[{prompt}] >>> ")
    sys.stdout.flush()

    rv = sys.stdin.readline().strip()

    if not rv:
        return None

    if rv.startswith("/"):
        rv = rv[1:].strip()
        tokens = shlex.split(rv)
        return Command(tokens[0], list(tokens[1:]))

    for speaker in speakers:
        assert speaker.affixes.prefix
        stripped_prefix = speaker.affixes.prefix.strip()
        if stripped_prefix and rv.startswith(stripped_prefix):
            rv = rv[len(stripped_prefix):].strip()
            return ChatInput(speaker, rv)
        
    for speaker in speakers:
        if speaker.human:
            return ChatInput(speaker, rv)

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

def choose_softmax(logits, temperature, exclude=set(), accept_prefixes: list[str] | None = None, text_so_far: str = ""):
    choices = []

    for logit in logits:
        if not is_acceptable_token(logit.token, exclude=exclude):
            continue

        token_str = logit.token.token_str.decode("utf-8")

        if accept_prefixes is not None and not remaining_options_for_prefixes(accept_prefixes, text_so_far + token_str):
            continue

        choices.append((math.exp(logit.logit / temperature), logit.token))

    if not choices:
        logging.fatal("No acceptable choices; logits were: %s", logits)

    total = sum([choice[0] for choice in choices])

    x = random.random() * total
    for choice in choices:
        x -= choice[0]
        if x <= 0:
            break

    return choice[1]

def remaining_options_for_prefixes(options, text_so_far):
    rv = []
    for opt in options:
        if opt.startswith(text_so_far) or text_so_far.startswith(opt):
            rv.append(opt)
    return rv

def generate_line(predict_with_extra_tokens, input_tokens: list[int], accept_speakers: dict[str, chatbot_pb2.Speaker], temperature: float):
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

    prefixes = [speaker.affixes.prefix for speaker in accept_speakers.values()]

    result = ""
    chosen_speaker = None

    generated_tokens = []

    while True:
        logging.debug(f"Feeding: {repr(input_tokens)} + {repr(generated_tokens)}")
        logits = predict_with_extra_tokens(extra_tokens=input_tokens + generated_tokens)
        if result.endswith("\n"):
            break
        exclude = {13} if not result else set()
        chosen_token = choose_softmax(logits, temperature=temperature, exclude=exclude, accept_prefixes=prefixes, text_so_far=result)
        logging.debug(f"Chose token: {chosen_token}")
        input_str = chosen_token.token_str.decode("utf-8")

        result += input_str
        generated_tokens.append(chosen_token.token_id)

        new_prefixes = remaining_options_for_prefixes(prefixes, result)
        assert new_prefixes
        prefixes = new_prefixes

        if chosen_speaker is None:
            if len(prefixes) == 1:
                possible_speakers = [speaker for speaker in accept_speakers.values() if speaker.affixes.prefix == prefixes[0]]
                if not possible_speakers:
                    raise RuntimeError(f"Prefix {prefixes[0]} is not a valid speaker prefix")
                chosen_speaker = possible_speakers[0]

                if chosen_speaker.affixes.suffix:
                    raise RuntimeError("Suffixes not yet supported")

    if chosen_speaker is None:
        raise RuntimeError(f"Could not determine speaker from prefixes {prefixes}")

    prefix = chosen_speaker.affixes.prefix
    suffix = chosen_speaker.affixes.suffix
    raw_result = result
    result = raw_result[len(prefix):len(raw_result)-len(suffix)].strip()

    return result, generated_tokens, chosen_speaker

def format_suitable_prompt(scenario, target_context_size, count_tokens):
    max_size = max_prompt_size(scenario)
    min_size = 2

    if max_size <= min_size:
        return format_initial_prompt(scenario, size=max_size)
    
    assert count_tokens(format_initial_prompt(scenario, size=min_size)) <= target_context_size


    lo = min_size
    hi = max_size + 1

    logging.info("min_size=%d, max_size=%d", min_size, max_size)

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

class ConversationChunk:
    def __init__(self, text: str, tokens: list[int], speaker: str | None, ignored: bool = False, stripped_text: str | None = None, provenance=None, labels=None):
        self.text = text
        self.tokens = tokens
        self.speaker = speaker
        self.ignored = ignored
        self.stripped_text = stripped_text or text
        self.provenance = provenance
        self.labels = labels or chatbot_pb2.Labels()
    
    def __repr__(self):
        return f"ConversationChunk(text={repr(self.text)}, tokens={repr(self.tokens)}, speaker={repr(self.speaker)}, ignored={repr(self.ignored)})"

class ConversationState:
    def __init__(self, fixed_prompt: ConversationChunk):
        self.fixed_prompt = fixed_prompt
        self.chunks = []
        self.undo_count = [0]

    def add_chunk(self, chunk: ConversationChunk):
        self.chunks.append(chunk)
        self.undo_count.append(0)

    def try_pop_chunk(self):
        if self.chunks:
            self.chunks.pop()
            self.undo_count.pop()
            self.undo_count[-1] += 1
            return True
        return False

    def get_number_of_attempts(self):
        return self.undo_count[-1] + 1

    def get_tokens(self) -> list[int]:
        tokens = list(self.fixed_prompt.tokens)
        for chunk in self.chunks:
            if not chunk.ignored:
                tokens.extend(chunk.tokens)
        return tokens

    def limit_context_size(self, max_tokens: int):
        budget = max_tokens - len(self.fixed_prompt.tokens)
        if budget < 0:
            raise RuntimeError("Fixed prompt is too long")
        for chunk in self.chunks:
            chunk.ignored = False

        started_ignoring = False
        for chunk in reversed(self.chunks):
            if (not started_ignoring) and budget >= len(chunk.tokens):
                budget -= len(chunk.tokens)
            else:
                started_ignoring = True
                chunk.ignored = True

    def print_summary(self):
        sys.stdout.write("---\n")
        sys.stdout.write(self.fixed_prompt.text)
        for chunk in self.chunks:
            if not chunk.ignored:
                sys.stdout.write(chunk.text)
    
    def proto_lines(self, exclude_ignored: bool = True):
        return [chatbot_pb2.Line(
            speaker=chunk.speaker,
            text=chunk.stripped_text,
            provenance=chunk.provenance,
            labels=chunk.labels if len(chunk.labels.percentile_rating) else None,
        ) for chunk in self.chunks if not (exclude_ignored and chunk.ignored)]

def main(argv):
  del argv  # Unused.

  load_from_filename = FLAGS.scenario
  checkpoint_suffix = ".checkpoint.pb_text"
  checkpoint_filename = load_from_filename + checkpoint_suffix
  if FLAGS.checkpoint_filename:
    checkpoint_filename = FLAGS.checkpoint_filename

  if FLAGS.recover_checkpoint:
    if os.path.exists(checkpoint_filename):
        logging.info(f"Loading checkpoint from {checkpoint_filename}")
        load_from_filename = checkpoint_filename

  with open(load_from_filename) as f:
    scenario = text_format.Parse(f.read(), chatbot_pb2.Scenario())
  
  logging.info(f"Connecting to: {FLAGS.server}")
  with grpc.insecure_channel(FLAGS.server) as channel:
    stub = llama_pb2_grpc.LlamaServiceStub(channel)

    session_hint = llama_pb2.SessionHint()

    req = llama_pb2.DoPredictRequest()
    req.model_info.model_name = FLAGS.model_name
    def tokenize(s):
        tokreq = llama_pb2.TokenizeRequest(text=s)
        tokreq.model_info.model_name = FLAGS.model_name
        tokreq.session_hint.CopyFrom(session_hint)
        response = stub.Tokenize(tokreq)
        return response.token

    def count_tokens(s):
        return len(tokenize(s))

    min_size = 2
    max_size = max_prompt_size(scenario)

    max_context_size = 2048
    max_fixed_prompt_size = max_context_size // 3

    scenario_without_chat = chatbot_pb2.Scenario()
    scenario_without_chat.CopyFrom(scenario)
    scenario_without_chat.setup.line.clear()

    prompt = format_suitable_prompt(scenario_without_chat, max_fixed_prompt_size, count_tokens)
    BOS = 1
    prompt_tokens = [BOS] + [token.token_id for token in tokenize(prompt)]

    speakers = parse_speakers(scenario.setup.context.speaker)

    convo = ConversationState(fixed_prompt=ConversationChunk(prompt, list(prompt_tokens), speaker=None))
    for line in scenario.setup.line:
        speaker = speakers[line.speaker]
        text = line.text
        formatted_line = f"{speaker.affixes.prefix}{text}{speaker.affixes.suffix}\n"
        tok = [token.token_id for token in tokenize(formatted_line)]
        convo.add_chunk(ConversationChunk(
            text=formatted_line,
            stripped_text=text,
            speaker=speaker.name,
            tokens=tok,
            provenance=line.provenance,
            labels=line.labels,
        ))

    tokens_slack = 256
    extra_budget = max_context_size - len(prompt_tokens) - tokens_slack
    assert extra_budget >= 0

    target_context_size = len(prompt_tokens) + extra_budget // 2
    context_size_threshold = len(prompt_tokens) + extra_budget

    req = llama_pb2.DoPredictRequest()
    req.model_info.model_name = FLAGS.model_name
    req.logit_processing.top_n = 40
    req.logit_processing.llama_repetition_penalty.intensity = 1.1

    convo.limit_context_size(target_context_size)
    req.full_context.token_ids.token_id[:] = convo.get_tokens()

    n_tokens = len(prompt_tokens)
    logging.info(f"Feeding: {repr(prompt)} [tokens: {n_tokens}]")

    t0 = time.time()
    response = stub.DoPredict(req)
    duration = time.time() - t0
    logging.info(f"Computed prompt of %d tokens in %.2f seconds", n_tokens, duration)

    req.session_hint.session_id = response.session_info.session_id

    for line in scenario.setup.line:
        speaker = speakers[line.speaker]
        print(f"{speaker.affixes.prefix}{line.text}{speaker.affixes.suffix}")
    sys.stdout.flush()

    metadata = ""

    def add_line(speaker, text, provenance=None):
        scenario.setup.line.add(speaker=speaker.name, text=text)
        formatted_line = f"{speaker.affixes.prefix}{text}{speaker.affixes.suffix}\n"
        sys.stdout.write(formatted_line)
        sys.stdout.flush()

        tok = [token.token_id for token in tokenize(formatted_line)]

        return ConversationChunk(
            text=formatted_line,
            stripped_text=text,
            speaker=speaker.name,
            tokens=tok,
            provenance=provenance,
        )
    
    def sync_scenario():
        prompt_tokens[:] = convo.get_tokens()
        scenario.setup.line.clear()
        scenario.setup.line.extend(convo.proto_lines(exclude_ignored=False))

        value = text_format.MessageToString(scenario)

        if FLAGS.output_transcript:
            tmp = f"{FLAGS.output_transcript}.tmp"
            with open(tmp, "w") as f:
                f.write(value)
            shutil.move(tmp, FLAGS.output_transcript)

        if FLAGS.write_checkpoint:
            tmp = f"{checkpoint_filename}.tmp"
            with open(tmp, "w") as f:
                f.write(value)
            shutil.move(tmp, checkpoint_filename)

    def predict_with_extra_tokens(extra_tokens):
        subreq = llama_pb2.DoPredictRequest()
        subreq.CopyFrom(req)
        subreq.full_context.token_ids.token_id[:] = prompt_tokens + extra_tokens
        response = stub.DoPredict(subreq)
        return response.next_token_logit

    alternatives = []

    def make_human_provenance():
        rv = chatbot_pb2.Provenance(
            provenance_type=chatbot_pb2.ProvenanceType.PROVENANCE_MANUAL,
            uuid=str(uuid.uuid4()),
            timestamp_unix_seconds=int(time.time()),
            attempt_count=convo.get_number_of_attempts(),
            discarded_alternative=list(alternatives),
        )
        alternatives.clear()
        return rv

    def make_generated_provenance(*, duration_seconds: float, temperature: float, partially_generated: bool = False):
        rv = chatbot_pb2.Provenance(
            provenance_type=chatbot_pb2.ProvenanceType.PROVENANCE_GENERATED if not partially_generated else chatbot_pb2.ProvenanceType.PROVENANCE_PARTIALLY_GENERATED,
            uuid=str(uuid.uuid4()),
            model_info=chatbot_pb2.ModelInfo(
                model_name=FLAGS.model_name,
            ),
            timestamp_unix_seconds=int(time.time()),
            generation_time_seconds=duration_seconds,
            attempt_count=convo.get_number_of_attempts(),
            temperature=temperature,
            discarded_alternative=list(alternatives),
        )
        alternatives.clear()
        return rv

    temperature = 0.5

    while True:
        bot_speakers = {speaker.name: speaker for speaker in speakers.values() if not speaker.human}
        human_speakers = [speaker for speaker in speakers.values() if speaker.human]
        assert len(human_speakers) <= 1

        input_tokens = []
        if human_speakers:
            while True:
                sync_scenario()

                sys.stdout.flush()
                input_ent = read_input(metadata, speakers=list(speakers.values()))
                sys.stdout.write("\033[1A\033[2K")
                match input_ent:
                    case None:
                        break
                    case ChatInput(speaker, input_line):
                        convo.add_chunk(add_line(speaker, input_line, provenance=make_human_provenance()))
                        if speaker.human:
                            break
                    case Command("back", []):
                        convo.try_pop_chunk()
                        line_proto_of_undone = chatbot_pb2.Line()
                        line_proto_of_undone.CopyFrom(scenario.setup.line[-1])
                        alternatives.append(line_proto_of_undone)
                        convo.print_summary()
                    case Command("addspeaker", [speaker_name]):
                        new_speaker = chatbot_pb2.Speaker(name=speaker_name)
                        fixed_speakers = parse_speakers([new_speaker])
                        speakers.update(fixed_speakers)
                        scenario.setup.context.speaker.append(new_speaker)
                    case Command("tokenize", [text]):
                        for token in tokenize(text):
                            print(token)
                    case Command("forceprefix", [text]):
                        # TODO this has the tokenization subtlety problem
                        input_tokens = [token.token_id for token in tokenize(text)]
                        break
                    case Command("set-temperature", [arg]):
                        new_temperature = float(arg)
                        logging.info(f"Changing temperature from {temperature} to {new_temperature}")
                        temperature = new_temperature
                    case Command("rate", [rating]):
                        if not rating.endswith("/100"):
                            print("Rating must be nn/100")
                            continue
                        rating = int(rating[:-4])
                        if rating < 0 or rating > 100:
                            print(f"Rating must be nn/100; {rating} is out of range")
                            continue
                        convo.chunks[-1].labels.percentile_rating.append(chatbot_pb2.PercentileRating(value=rating))
                    case Command(cmd, _):
                        print(f"Unknown command: {cmd}")
                    case _:
                        sys.stdout.write("???\n")
        sys.stdout.flush()
        sync_scenario()

        sys.stdout.write("...\n")
        sys.stdout.flush()

        force_prefix = input_tokens

        t0 = time.time()
        output_line, output_tokens, chosen_speaker = generate_line(
            predict_with_extra_tokens,
            input_tokens=force_prefix,
            accept_speakers=bot_speakers,
            temperature=temperature,
        )
        duration = time.time() - t0
        duration_ms = int(duration * 1000)

        sys.stdout.write("\033[1A\033[2K")
        sys.stdout.flush()

        convo.add_chunk(add_line(chosen_speaker, output_line, provenance=make_generated_provenance(
            partially_generated=bool(force_prefix),
            duration_seconds=duration,
            temperature=temperature,
        )))
        sync_scenario()

        current_context_size = len(prompt_tokens)

        logging.debug(f"context_size=%d duration_ms=%d", current_context_size, duration_ms)

        metadata = f"[{current_context_size} {duration_ms}] "

        if current_context_size > context_size_threshold:
            logging.info("Threshold reached; context is %d tokens", len(prompt_tokens))
            convo.limit_context_size(target_context_size)
            prompt_tokens[:] = convo.get_tokens()
            logging.info("Recomputed context size is now %d tokens", len(prompt_tokens))

if __name__ == '__main__':
  app.run(main)
