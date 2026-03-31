# Changelog

If you want to read my dribble as I update this code.  Go on.

## Initial Commit
### **1.0.0** - ``2024-07-18``

- Move to own repo/branch, fork for Furry Diffusion community and begin rebranding repository.
- Posting memes

--------------

## Update: Mlem
### **1.0.1** - ``2024-07-19``

- Listing on Comfy-UI registry, some code clean up and corrections, port in gradio inference code for RedRocket.
- Mlemming starts.

--------------

## Update: Get it away!
### **1.0.2** ``2024-07-19``

- What in the fuck was this shit?  I stayed up all night and it was complete shit.
- logbits, probits, I missed a few things in translation.  Those got fixed.
- Passing through version params weirdly fixed.
- Standing on a mattress screaming at the bugs.

--------------

## Update: What's sleep?
### **1.1.0** - ``2024-07-21``

- Big refactor of all of the code... exhausting.
- Modularization of everything, including the ComfyUI extension helpers themselves.
- Use orjson for speedup on load tags.json.
- Add model and tag tensor/object caching with ``ComfyCache`` class.
- Add model and tag loading and unloading for memory efficiency with ``JtpTagManager``, ``JtpModelManager``, ``JtpImageManager`` classes.
- Improve this chamgelog format slightly.
- Add code comments and lint some code manualy with autopep8.  Not complete.
- Touched some Grass, whips and chains.

--------------

## Update: Wildcard Exclusions
### **1.2.0** - ``2026-03-31``

- Add glob-style `*` wildcard support to the `exclude_tags` input across all tagger variants (V1/V2, V3, DINOv3).
- Wildcards work in any position: `human*` (starts with), `*human` (ends with), `*human*` (contains), `human*top` (starts/ends), and any combination.
- Underscores in patterns are normalised to spaces as before — e.g. `anthro_*` and `anthro *` behave identically.
- Tags without `*` continue to work as exact matches (fully backward compatible).

--------------