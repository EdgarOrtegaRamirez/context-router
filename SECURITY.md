# Security

## Security Policy

### API Key Management
- **Never** commit API keys or tokens to version control
- Use environment variables or `.env` files (added to `.gitignore`)
- The project uses `pydantic-settings` for secure config loading

### Input Validation
- All user inputs are validated using Pydantic models
- Prompt text is sanitized before processing
- File paths are validated to prevent path traversal

### Network Security
- HTTPS-only connections to API endpoints
- No certificate bypass
- Connection timeouts configured to prevent hanging

### Dependency Security
- All dependencies are version-pinned in `pyproject.toml`
- No use of `latest` or wildcard versions
- Regular dependency updates via CI

### Reporting
If you discover a security vulnerability, please open a GitHub issue with the label "security".
