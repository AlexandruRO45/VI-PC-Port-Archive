## Summary

## Checks

CI runs the Python unit tests, the shader compiler, DXBC, and XG layout tests, and a Windows typecheck of `tuxbox-d3d12`.

If this pull request changes evidence JSON, regenerate the results table and the progress map and include both files:

```sh
python tuxbox.py status > docs/RESULTS.md
python tuxbox.py status --svg docs/progress.svg
```
