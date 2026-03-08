class AppError(Exception):
    """Base application error."""

    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BadRequestError(AppError):
    status_code = 400


class NotFoundError(AppError):
    status_code = 404


class ExternalServiceError(AppError):
    status_code = 502
