install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall torchtree_bito

lint: FORCE
	flake8 --exit-zero torchtree_bito test
	isort --check .
	black --check .

format: license FORCE
	isort .
	black .

test: FORCE
	pytest

clean: FORCE
	rm -fr torchtree_bito/__pycache__ build var

nuke: FORCE
	git clean -dfx -e torchtree_bito.egg-info

	done

FORCE: