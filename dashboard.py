# Run this app with `python dashboard.py` and
# Visit http://127.0.0.1:8050/ in your web browser.

# Imports
from engine.engine import Engine


class Dashboard:
    """Wrapper class for Engine."""
    engine = Engine()

    @classmethod
    def run(dashboard, *args, **kwargs) -> None:
        dashboard.engine.app.run_server(*args, **kwargs)

    # See "Flexible Callback Signatures" in dash documentation
    @engine.app.callback(inputs={"inputs": engine.update_state_cb_inputs}, output={"outputs": engine.update_state_cb_outputs})
    def update_state(inputs):
        return {"outputs": Dashboard.engine.update_state(inputs)}

    @engine.app.callback(inputs={"inputs": engine.graph_cb_inputs}, output={"outputs": engine.graph_cb_outputs})
    def graph_callback(inputs):
        return {"outputs": Dashboard.engine.graph_callback(inputs)}


if __name__ == "__main__":
    Dashboard.run(debug=Engine.settings.DEBUG, port=Engine.settings.port_number)
