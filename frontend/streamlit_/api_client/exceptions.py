class APIError(Exception):
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class SessionNotFoundError(APIError):
    def __init__(self, session_id: str):
        super().__init__(f"Session not found: {session_id}", 404)


class AgentNotFoundError(APIError):
    def __init__(self, session_id: str):
        super().__init__(f"Agent not found for session: {session_id}", 404)
