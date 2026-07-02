PY := .venv/bin/python
PYTEST := .venv/bin/pytest
SCENE ?= billiards

.PHONY: test integ data train grid render site demo

test:
	$(PYTEST) tests -m "not integ" -q

integ:
	$(PYTEST) tests -m integ -q

data:
	$(PY) -m wa.data --scene $(SCENE)

train:
	$(PY) -m wa.models --scene $(SCENE)

grid:
	$(PY) -m wa.grid --scene $(SCENE)

render:
	MUJOCO_GL=egl $(PY) -m wa.render

site:
	cd site && npm run build

namebrand:
	MUJOCO_GL=egl $(PY) -m wa.namebrand.audit

demo:
	$(MAKE) data SCENE=$(SCENE) && $(MAKE) train SCENE=$(SCENE) && $(MAKE) grid SCENE=$(SCENE) && $(MAKE) render && $(MAKE) site
