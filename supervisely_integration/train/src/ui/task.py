from supervisely.app.widgets import Card, Container, RadioGroup, Button, NotificationBox, Field

from src.ui.utils import update_custom_button_params

msg = """
    <b>Object detection</b>: the model will predict bounding boxes of the objects.
    All annotations will be converted to Rectangles.
    <br>
    <b>Instance Segmentation</b>: the model will predict bounding boxes and masks of the objects.
    Only Bitmap and Polygon annotations will be used."""

info = NotificationBox(title="INFO: How to select the task?", description=msg, box_type="info")
task_selector = RadioGroup(
    items=[
        RadioGroup.Item(value="Object detection", label="Object detection"),
        RadioGroup.Item(value="Instance segmentation", label="Instance segmentation"),
    ],
    direction="vertical",
)


select_field = Field(title="Select the task you are going to solve:", content=task_selector)
select_btn = Button(text="Select task")

card = Card(
    title="MMDetection task",
    description="Select task from list below",
    content=Container(widgets=[info, select_field, select_btn], direction="vertical"),
    lock_message="Please, select project and load data.",
)
