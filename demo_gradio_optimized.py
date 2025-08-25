# Optimized entry: uses VGGTOptimized with selected_layer_idx=[4,11,17,23]
# to reduce returned intermediate features without modifying original demo.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt_vram_optimized import VGGTOptimized as VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model (optimized)...")
model = VGGT(selected_layer_idx=[4, 11, 17, 23])
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)


def run_model(target_dir, model) -> dict:
    print(f"Processing images from {target_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    model = model.to(device)
    model.eval()

    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Debug: timing + peak CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
    t1 = time.perf_counter()
    print(f"[Debug] Inference time: {t1 - t0:.3f}s")
    if torch.cuda.is_available():
        print(f"[Debug] Peak CUDA memory: {peak_mem/1e6:.2f} MB")

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions['pose_enc_list'] = None

    print("Computing world points from depth map...")
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    torch.cuda.empty_cache()
    return predictions


def handle_uploads(input_video, input_images):
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


def update_gallery_on_upload(input_video, input_images):
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No reconstruction available. Please click the Reconstruct button first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    if frame_filter is None:
        frame_filter = "All"

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


def clear_fields():
    return None


def update_log():
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(theme=theme) as demo:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses) — Optimized**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.")

            with gr.Row():
                reconstruct_button = gr.Button("Reconstruct", variant="primary")
                clear_button = gr.Button("Clear")

            glb_viewer = gr.Model3D(label="3D Viewer", clear_color=[1.0, 1.0, 1.0, 0.0])
            glb_output = gr.File(label="Download GLB")

            with gr.Row():
                conf_thres = gr.Slider(value=3.0, minimum=0.0, maximum=20.0, step=0.5, label="Confidence Threshold")
                show_cam = gr.Checkbox(value=True, label="Show Camera")
                mask_black_bg = gr.Checkbox(value=False, label="Filter Black Background")
                mask_white_bg = gr.Checkbox(value=False, label="Filter White Background")
                mask_sky = gr.Checkbox(value=False, label="Filter Sky")

            frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frames", interactive=True)
            prediction_mode = gr.Radio(
                choices=["Depthmap and Camera Branch", "Pointmap Branch"],
                value="Pointmap Branch",
                label="Prediction Mode",
                elem_id="my_radio",
            )

    # ---------------------- Examples section ----------------------
    great_wall_video = "examples/videos/great_wall.mp4"
    colosseum_video = "examples/videos/Colosseum.mp4"
    room_video = "examples/videos/room.mp4"
    kitchen_video = "examples/videos/kitchen.mp4"
    fern_video = "examples/videos/fern.mp4"
    single_cartoon_video = "examples/videos/single_cartoon.mp4"
    single_oil_painting_video = "examples/videos/single_oil_painting.mp4"
    pyramid_video = "examples/videos/pyramid.mp4"

    examples = [
        [colosseum_video, "22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [pyramid_video, "30", None, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_cartoon_video, "1", None, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_oil_painting_video, "1", None, 20.0, False, False, True, True, "Depthmap and Camera Branch", "True"],
        [room_video, "8", None, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [kitchen_video, "25", None, 50.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [fern_video, "20", None, 45.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
    ]

    def example_pipeline(
        input_video,
        num_images_str,
        input_images,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        is_example_str,
    ):
        target_dir, image_paths = handle_uploads(input_video, input_images)
        frame_filter_val = "All"
        glbfile, log_msg, dropdown = gradio_demo(
            target_dir, conf_thres, frame_filter_val, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
        )
        return glbfile, log_msg, target_dir, dropdown, image_paths

    gr.Markdown("Click any row to load an example.")
    gr.Examples(
        examples=examples,
        inputs=[
            input_video,
            num_images,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        outputs=[glb_viewer, log_output, target_dir_output, frame_filter, image_gallery],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    # Submit button chain similar to original
    reconstruct_button.click(fn=clear_fields, inputs=[], outputs=[glb_viewer]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        outputs=[glb_viewer, log_output, frame_filter],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]
    )

    # Real-time visualization updates
    for comp in [conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode]:
        comp.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            [glb_viewer, log_output],
        )

    # Auto-update gallery on upload
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[glb_viewer, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[glb_viewer, target_dir_output, image_gallery, log_output],
    )

demo.queue(max_size=32)
demo.launch(server_name="0.0.0.0", server_port=8080, share=False)
