def chunk_by_sentences(input_text: str, tokenizer: callable):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == punctuation_mark_id
        and (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
        )
    ]
    # for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets)):
    #     print(start, end, input_text[start:end])
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations


def chunk_by_config(input_text: str, tokenizer: callable, chunk_sizes: list[int]):
    """
    Split the input text into chunks based on a list of chunk size
    :param input_text: The text snippet to split into chunks
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    chunk_sizes = [chunk_sizes[0]] + [chunk_sizes[i] + chunk_sizes[i - 1] for i in range(1, len(chunk_sizes))]
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if int(start + 1) in chunk_sizes
    ]
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations


def chunk_by_config2(input_text: str, tokenizer: callable, chunk_contents: list[str]):
    """
    Split the input text into chunks based on a list of chunk contents
    :param input_text: The text snippet to split into chunks
    :param tokenizer: The tokenizer to use
    :param chunk_contents: The list of chunk contents to use for splitting
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    chunk_sizes = [len(x) for x in chunk_contents]
    # chunk_sizes = [chunk_sizes[0]] + [chunk_sizes[i] + chunk_sizes[i - 1] for i in range(1, len(chunk_sizes))]
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = []
    t_start = 0
    for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets)):
        if int(end) - t_start == chunk_sizes[len(chunk_positions)]:
            chunk_positions.append((i, int(start + 1)))
            if i < len(token_offsets) - 1:
                t_start = int(token_offsets[i + 1][0])
        if len(chunk_positions) == len(chunk_sizes):
            break
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations


def chunked_pooling(
    model_output: 'BatchEncoding', span_annotation: list, max_length=None
):
    token_embeddings = model_output[0]
    outputs = []

    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.float().detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs


def chunked_long_latechunking_pooling(
    model_output: 'BatchEncoding', span_annotation: list, max_length=None, long_late_chunking=False
):
    token_embeddings = model_output[0]
    outputs = []

    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.float().detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs
