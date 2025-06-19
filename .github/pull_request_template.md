# Pull Request

## Description
<!-- Provide a brief description of the changes in this PR -->

## Type of Change
<!-- Mark the relevant option with an "x" -->
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Maintenance (dependency updates, CI improvements, etc.)
- [ ] 🎨 Code style/formatting changes
- [ ] ⚡ Performance improvement

## Related Issues
<!-- Link to any related issues -->
Fixes #(issue number)
Relates to #(issue number)

## Changes Made
<!-- List the specific changes made in this PR -->
- 
- 
- 

## Testing
<!-- Describe the tests you ran and how to reproduce them -->
- [ ] Unit tests pass (`pytest tests/unit/`)
- [ ] Integration tests pass (`pytest tests/e2e/`)
- [ ] Performance tests pass (`pytest tests/performance/`)
- [ ] Manual testing completed
- [ ] New tests added for new functionality

### Test Commands
```bash
# Commands used to test these changes
pytest tests/unit/test_specific_module.py -v
```

## Documentation
<!-- Mark if documentation was updated -->
- [ ] Code comments updated
- [ ] README updated
- [ ] API documentation updated
- [ ] CLAUDE.md updated
- [ ] No documentation changes needed

## Deployment
<!-- For deployment-related changes -->
- [ ] Docker images build successfully
- [ ] Kubernetes manifests validated
- [ ] Staging deployment tested
- [ ] Migration scripts provided (if needed)
- [ ] Rollback plan documented

## Checklist
<!-- Mark completed items with an "x" -->
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Screenshots
<!-- If applicable, add screenshots to help explain your changes -->

## Additional Notes
<!-- Add any additional notes for reviewers -->

---

## For Reviewers
<!-- This section is for reviewer guidance -->

### Review Focus Areas
- [ ] Code quality and maintainability
- [ ] Test coverage and quality
- [ ] Performance implications
- [ ] Security considerations
- [ ] Documentation completeness
- [ ] Breaking change assessment

### Deployment Checklist
- [ ] Docker images can be built
- [ ] Kubernetes manifests are valid
- [ ] Environment variables are documented
- [ ] Migration steps are clear (if applicable)