import sys, time, traceback, tkinter as tk
from PIL import Image
errors = []
import studio_app
root = tk.Tk()
root.report_callback_exception = lambda *a: errors.append("".join(traceback.format_exception(*a)))
app = studio_app.VHRProStudioApp(root)

def spin(sec):
    end = time.time() + sec
    while time.time() < end:
        root.update(); time.sleep(0.005)

def shot(name):
    root.update(); spin(0.2)
    if app.current_rgb is not None:
        Image.fromarray(app.current_rgb).save(f"scratch/{name}.png")
        print("saved frame", name, app.current_rgb.shape)

root.geometry("1620x980+20+20"); spin(2.0)
print("1. equation tab status:", app.eq_msg_var.get(), "| telemetry:", app.telemetry_var.get().splitlines()[0])
shot("ui_equation")

# every example must compile and render
for name in studio_app.EXAMPLES:
    app._load_example(name); spin(1.2)
    print(f"   example {name!r}: {app.eq_msg_var.get()} | {app.telemetry_var.get().splitlines()[0]}")
app._load_example("Spiral galaxy (sum / prod)"); spin(2.0); shot("ui_galaxy_eq")

# slider edit through the editor text
app._load_example("Plasma (animated)"); spin(1.0)
spec = app._slider_specs[1]; app._write_slider_value(spec.line_index, 6.0); spin(1.0)
print("2. slider wrote:", app.editor.get(f"{spec.line_index+1}.0", f"{spec.line_index+1}.end"))

# error handling
app.editor.insert("end", "\nH = nonsense(x)\n"); app._on_source_edit(immediate=True); spin(0.6)
print("3. error message shown:", app.eq_msg_var.get())
app._load_example("Plasma (animated)"); spin(0.8)

# animation
app.anim_var.set(True); app._toggle_animation(); spin(2.5)
print("4. animation:", app.fps_var.get()); shot("ui_anim"); app.anim_var.set(False); app._toggle_animation(); spin(0.5)

# image -> equation
app.tabs.select(1); spin(0.3)
app._set_source(path="scratch/test_4x3.png"); spin(0.5)
app.max_side_var.set("320"); app.run_conversion()
t0 = time.time()
while app.analysis is None and time.time() - t0 < 60: spin(0.1)
spin(1.5)
print("5. convert:", app.k_label_var.get()); print(app.telemetry_var.get())
shot("ui_image")
for pos in (300, 500, 800, 1000):
    app.k_pos_var.set(pos); t1 = time.time(); app._on_k_slider(); spin(0.5)
    print(f"   K slider {pos}: {app.k_label_var.get()} -> {app.telemetry_var.get().splitlines()[0]}")
app.view_mode_var.set("wipe"); app.viewport.set_mode("wipe"); spin(0.2)
app.view_mode_var.set("error"); app.viewport.set_mode("error"); spin(0.2)
app.view_mode_var.set("layers"); app.viewport.set_mode("layers"); spin(0.2)
app.view_mode_var.set("result"); app.viewport.set_mode("result")

# send to editor & verify equation text renders to the same picture
app.send_to_editor(); spin(3.5)
print("6. sent to editor:", app.eq_msg_var.get(), "|", app.telemetry_var.get().splitlines()[0])
shot("ui_sent_to_editor")

# catalog
app.tabs.select(2); spin(0.3); app.catalog_var.set("galaxy_cluster"); app.gal_p_var.set(20); app.star_k_var.set(30); app.run_exact_catalog()
t0 = time.time()
while app.context != "catalog" and time.time() - t0 < 60: spin(0.1)
spin(0.5); print("7. catalog:", app.telemetry_var.get().replace("\n", " | ")); shot("ui_catalog")
app.tabs.select(0); spin(0.5)
print("TK CALLBACK ERRORS:", len(errors))
for e in errors[:3]: print(e)
root.destroy()
