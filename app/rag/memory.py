from collections import defaultdict


_session_memory = defaultdict(list)


def get_history(session_id: str):
    return _session_memory[session_id]


def add_message(session_id: str, role: str, content: str):
    _session_memory[session_id].append(f"{role}: {content}")


def clear_history(session_id: str):
    _session_memory[session_id].clear()