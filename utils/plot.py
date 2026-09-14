import matplotlib.pyplot as plt
import matplotlib.patches as pt
from matplotlib.colors import to_hex

def make_color_code(id_to_name):
    n = len(id_to_name)
    cmap = plt.cm.get_cmap('tab20', n)  # tab20 has 20 distinct colors, cycles/interpolates beyond that
    return {cat_id: to_hex(cmap(i)) for i, cat_id in enumerate(id_to_name.keys())}

color_code = make_color_code(id_to_name)

def visualize(
    id_to_name,
    color_code,
    img,
    preds,
    scores=None,
    labels=None
):
   """ Visualize the image with boxes and labels, show scores if scores provided """

   # Convert tensor image to numpy for matplotlib
   img_np = img.permute(1, 2, 0).cpu().numpy()

   # Get image dimensions
   H, W, _ = img_np.shape

   fig, ax = plt.subplots(1)
   ax.imshow(img_np)

   # Iterate over bounding boxes
   for i, box in enumerate(preds):
       # Convert normalized xyxy to pixel xyxy
       x_min, y_min, x_max, y_max = box.cpu().numpy()
       x_min *= W
       y_min *= H
       x_max *= W
       y_max *= H

       # Convert xyxy to xywh for matplotlib.patches.Rectangle
       rect_x = x_min
       rect_y = y_min
       rect_w = x_max - x_min
       rect_h = y_max - y_min

       color = color_code[labels[i].item()] if labels is not None and labels[i].item() in color_code else 'blue'

       # Create a Rectangle patch
       rect = pt.Rectangle(
           (rect_x,
            rect_y),
           rect_w,
           rect_h,
           linewidth=2,
           edgecolor=color,
           facecolor='none'
       )

       # Add the patch to the Axes
       ax.add_patch(rect)

       # Add score text if provided
       if scores is not None:
           score = scores[i].item()
           label_text = f"{id_to_name.get(labels[i].item(), 'Unknown')}: {score:.2f}" if labels is not None else f"{score:.2f}"
           ax.text(
               rect_x,
               rect_y - 10, # Position text slightly above the box
               label_text,
               color='white',
               fontsize=8,
               bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1)
           )

   ax.set_axis_off()
   plt.show()