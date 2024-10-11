from supervisely.app.widgets import ClassesTable, Card, Container, Button, Switch, Field
from supervisely.app.content import StateJson

from src.sly_globals import PROJECT_ID


def select_all(cls_tbl: ClassesTable):
    cls_tbl._global_checkbox = True
    cls_tbl._checkboxes = [True] * len(cls_tbl._table_data)
    StateJson()[cls_tbl.widget_id]["global_checkbox"] = cls_tbl._global_checkbox
    StateJson()[cls_tbl.widget_id]["checkboxes"] = cls_tbl._checkboxes
    StateJson().send_changes()


classes = ClassesTable(project_id=PROJECT_ID)
select_all(classes)

filter_images_without_gt_input = Switch(True)
filter_images_without_gt_field = Field(
    filter_images_without_gt_input,
    title="Filter images without annotations",
    description="After selecting classes, some images may not have any annotations. Whether to remove them?",
)

select_btn = Button("Select")
card = Card(
    title="Training classes",
    description=(
        "Select classes that will be used for training. "
        "Supported shapes are Bitmap, Polygon, Rectangle."
    ),
    content=Container([classes, filter_images_without_gt_field, select_btn]),
)
card.lock("Select model to unlock.")

# @classes.value_changed
# def confirmation_message(selected_classes):
#     selected_num = len(selected_classes)
#     if selected_num == 0:
#         select_btn.disable()
#         select_btn.text = "Select classes"
#     else:
#         select_btn.enable()
#         select_btn.text = f"Use {selected_num} selected classes"
