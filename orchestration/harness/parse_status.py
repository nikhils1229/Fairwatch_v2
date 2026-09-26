"""Parse status constants and canonical vocabulary for FairWatch."""

PARSE_OK = "ok"
PARSE_REPAIRED_NULL = "repaired_null"
PARSE_REPAIRED_FUZZY = "repaired_fuzzy"
PARSE_FALLBACK = "fallback"
PARSE_UNPARSEABLE = "unparseable"
PARSE_ERROR = "error"

# Extended validity classifications for analysis & legacy segregation
PARSE_LEGACY_OK = "legacy_ok"
PARSE_LEGACY_SYSTEM_ERROR = "legacy_system_error"
PARSE_FUZZY_FALLBACK = "fuzzy_fallback"
PARSE_TEMPLATE_COPY = "template_copy"
PARSE_INVALID_DECISION = "invalid_decision"

ALL_PARSE_STATUSES = {
    PARSE_OK,
    PARSE_REPAIRED_NULL,
    PARSE_REPAIRED_FUZZY,
    PARSE_FALLBACK,
    PARSE_UNPARSEABLE,
    PARSE_ERROR,
    PARSE_LEGACY_OK,
    PARSE_LEGACY_SYSTEM_ERROR,
    PARSE_FUZZY_FALLBACK,
    PARSE_TEMPLATE_COPY,
    PARSE_INVALID_DECISION,
}
