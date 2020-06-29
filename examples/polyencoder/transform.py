from itertools import chain


class SelectionSequentialTransform(object):
    def __init__(self, tokenizer, max_seq_len=128, max_history=10, pair_last=False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_history = max_history
        self.pair_last = pair_last

    def __call__(self, texts):
        input_ids_list, segment_ids_list, input_masks_list, contexts_masks_list = (
            [],
            [],
            [],
            [],
        )
        if self.max_history is not None:
            texts = texts[-self.max_history :]

        last_context = None
        if self.pair_last:
            last_context = texts[-1]

        for text in texts:
            tokenized_dict = self.tokenizer.encode_plus(
                text,
                text_pair=last_context,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                pad_to_max_length=True,
            )
            input_ids, segment_ids, input_masks = (
                tokenized_dict["input_ids"],
                tokenized_dict["token_type_ids"],
                tokenized_dict["attention_mask"],
            )

            # assert len(input_ids) == self.max_seq_len
            # assert len(segment_ids) == self.max_seq_len
            # assert len(input_masks) == self.max_seq_len

            input_ids_list.append(input_ids)
            segment_ids_list.append(segment_ids)
            input_masks_list.append(input_masks)

        contexts_masks_list = [1] * len(input_ids_list)

        if self.max_history is not None:
            tokenized_dict = self.tokenizer.encode_plus(
                "",
                text_pair="",
                add_special_tokens=True,
                max_length=self.max_seq_len,
                pad_to_max_length=True,
            )
            input_ids, segment_ids, input_masks = (
                tokenized_dict["input_ids"],
                tokenized_dict["token_type_ids"],
                tokenized_dict["attention_mask"],
            )
            for _ in range(self.max_history - len(texts)):
                input_ids_list.append(input_ids[:])
                segment_ids_list.append(segment_ids[:])
                input_masks_list.append(input_masks[:])

            contexts_masks_list += [0] * (self.max_history - len(texts))

        return input_ids_list, segment_ids_list, input_masks_list, contexts_masks_list


class SelectionJoinTransform(object):
    def __init__(self, tokenizer, max_seq_len=512, max_history=10):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_history = max_history

        self.cls_id = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.sep_id = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        self.pad_id = 0

    def __call__(self, texts):
        input_ids_list, segment_ids_list, input_masks_list = [], [], []

        total_seq_size = 0

        for text in texts[::-1][: self.max_history]:  # 优先保证最后一个context的信息量
            tokenized_dict = self.tokenizer.encode_plus(
                text,
                text_pair=None,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                pad_to_max_length=False,
            )
            input_ids, input_masks = (
                tokenized_dict["input_ids"],
                tokenized_dict["attention_mask"],
            )
            segment_ids = [0] * len(input_ids)

            seq_len = self.max_seq_len - total_seq_size
            if seq_len < len(input_ids) and total_seq_size > 0:
                break

            if len(input_ids_list) > 0:  # ignore CLS token
                input_ids = input_ids[1 : seq_len + 1]
                segment_ids = segment_ids[1 : seq_len + 1]
                input_masks = input_masks[1 : seq_len + 1]

            input_ids_list.append(input_ids)
            segment_ids_list.append(segment_ids)
            input_masks_list.append(input_masks)

            total_seq_size += len(input_ids)

            if total_seq_size >= self.max_seq_len:
                break

        # reverse the context order
        input_ids = list(chain(*input_ids_list[::-1]))
        segment_ids = list(chain(*segment_ids_list[::-1]))
        input_masks = list(chain(*input_masks_list[::-1]))

        # input_ids += [self.pad_id] * (self.max_len - len(input_ids))
        # segment_ids_list += [0] * (self.max_len - len(segment_ids))
        # input_masks_list += [0] * (self.max_len - len(input_masks))

        # assert len(input_ids_list) == self.max_seq_len
        # assert len(segment_ids_list) == self.max_seq_len
        # assert len(input_masks_list) == self.max_seq_len

        return input_ids, segment_ids, input_masks
