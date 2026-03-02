def extract_answer_span(context, question):
    """
    Extracts the answer span from the context based on the question.
    
    Parameters:
    context (str): The context from which to extract the answer.
    question (str): The question pertaining to the context.
    
    Returns:
    tuple: (start_index, end_index) of the answer span, or (-1, -1) if not found.
    """
    # Here you would implement the logic to find the answer span
    # For now, we are using a placeholder implementation
    start_index = context.find(question)
    if start_index != -1:
        end_index = start_index + len(question)
        return (start_index, end_index)
    return (-1, -1)