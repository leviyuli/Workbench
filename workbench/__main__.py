"""Entry point: ``python -m workbench``."""

# Spawned workers import this module without launching the GUI.
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    from workbench.app import main
    main()
