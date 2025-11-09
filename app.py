from flask import Flask, render_template, request
import importlib
import traceback

app = Flask(__name__)

# We'll import backend modules lazily to avoid heavy imports (pandas/sklearn) at app startup.
_loaded = {}


def get_module(name):
    """Lazily import and cache a backend module by short name (e.g. 'project1')."""
    if name in _loaded:
        return _loaded[name]
    try:
        mod = importlib.import_module(f'codtech_backend.{name}')
        _loaded[name] = mod
        return mod
    except Exception:
        print(f"Error importing codtech_backend.{name}:\n" + traceback.format_exc())
        raise


def ensure_trained(module):
    """Train the module's model if not already trained."""
    try:
        if getattr(module, 'model', None) is None:
            print(f"Training {module.__name__}...")
            module.train_model()
            print(f"Trained {module.__name__}.")
    except Exception:
        print(f"Error training {module.__name__}:\n" + traceback.format_exc())
        raise


@app.route('/')
def index():
    return render_template('index.html')


def handle_project(module_name, template_name, req):
    module = get_module(module_name)
    # train on first access so app startup is quick
    ensure_trained(module)

    if req.method == 'POST':
        form = req.form.to_dict()
        try:
            res = module.predict(form)
            metrics = res.get('metrics', {}) if isinstance(res, dict) else {}
            return render_template(template_name, result=res, form=form, metrics=metrics)
        except Exception as e:
            return render_template(template_name, result={'error': str(e)}, form=form, metrics={})
    else:
        return render_template(template_name, result=None, form=None, metrics={})


@app.route('/project1', methods=['GET', 'POST'])
def project1_route():
    return handle_project('project1', 'project1.html', request)


@app.route('/project2', methods=['GET', 'POST'])
def project2_route():
    return handle_project('project2', 'project2.html', request)


@app.route('/project3', methods=['GET', 'POST'])
def project3_route():
    return handle_project('project3', 'project3.html', request)


@app.route('/project4', methods=['GET', 'POST'])
def project4_route():
    return handle_project('project4', 'project4.html', request)


if __name__ == '__main__':
    app.run(debug=True)
