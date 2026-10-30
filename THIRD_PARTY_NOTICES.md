# Third-party notices

The Windows folder distribution bundles third-party software and model
weights. This summary identifies principal components; it is not legal advice
or a substitute for their complete license texts, notices, and redistribution
conditions. Review the licenses in the bundled `.dist-info\licenses` folders
and the linked upstream terms before redistributing.

| Component | License / terms | Source |
|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M weights | Apache-2.0 | [Model repository](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) |
| llama.cpp | MIT | [llama.cpp repository](https://github.com/ggml-org/llama.cpp) |
| llama-cpp-python | MIT | [llama-cpp-python repository](https://github.com/abetlen/llama-cpp-python) |
| PyMuPDF | AGPL-3.0 or commercial license | [PyMuPDF licensing](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright) |
| PySide6 / Qt and Shiboken | LGPL-3.0-only, GPL-2.0-only, GPL-3.0-only, or commercial terms, depending on use | [Qt licensing](https://www.qt.io/licensing/) |
| scikit-learn | BSD-3-Clause | [scikit-learn repository](https://github.com/scikit-learn/scikit-learn) |
| NumPy | BSD-3-Clause | [NumPy repository](https://github.com/numpy/numpy) |
| SciPy | BSD-3-Clause | [SciPy repository](https://github.com/scipy/scipy) |
| Python runtime | Python Software Foundation License | [Python license](https://docs.python.org/3/license.html) |
| PyInstaller | GPL with a bootloader exception | [PyInstaller license](https://pyinstaller.org/en/stable/license.html) |

The bundled llama.cpp build may also include GNU and Microsoft runtime
libraries. Preserve the notices and redistribution terms supplied with those
native components. The project has not received a legal review or selected a
commercial PyMuPDF/Qt license; the current folder build is for technical
validation and must not be published until distribution rights are confirmed.
