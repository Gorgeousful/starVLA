import json
from pathlib import Path
from typing import Any

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


CUSTOM_TASK_SEGMENT_FILENAME = "meta/tasks_segment.json"
CUSTOM_GROUNDING_FILENAME = "meta/episodes_grounding.jsonl"
CUSTOM_SUBTASK_SEGMENT_FILENAME = "meta/episodes_subtask_segment.jsonl"
CUSTOM_PHASE_SEGMENT_FILENAME = "meta/episodes_phase_segment.jsonl"


class CustomAnnotationIndex:
    def __init__(
        self,
        dataset_path: Path,
        dataset_fps: float | int | None = None,
        include_gripper: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_fps = float(dataset_fps) if dataset_fps is not None else None
        self.include_gripper = include_gripper

        self.task_segments = self._load_task_segments()
        self.grounding_by_episode = self._load_jsonl_by_episode(CUSTOM_GROUNDING_FILENAME)
        self.subtasks_by_episode = self._load_jsonl_by_episode(CUSTOM_SUBTASK_SEGMENT_FILENAME)
        self.phases_by_episode_subtask = self._load_phase_segments()

    def get_sample_fields(self, episode_index: int, base_index: int) -> dict[str, Any]:
        episode_index = int(episode_index)
        base_index = int(base_index)

        grounding_record = self._get_required(
            self.grounding_by_episode,
            episode_index,
            f"grounding record for episode_index={episode_index}",
        )
        self._check_fps(grounding_record, "grounding", episode_index)

        subtask_record = self._get_required(
            self.subtasks_by_episode,
            episode_index,
            f"subtask segment record for episode_index={episode_index}",
        )
        self._check_fps(subtask_record, "subtask", episode_index)

        subtask_segment = self._find_segment(
            subtask_record.get("segments"),
            base_index,
            f"subtask for episode_index={episode_index}, base_index={base_index}",
        )
        subtask_id = str(subtask_segment.get("sub_task_id", "")).strip()
        if not subtask_id:
            raise ValueError(
                f"Missing sub_task_id for episode_index={episode_index}, base_index={base_index}"
            )

        task_index = int(subtask_record.get("task_index", grounding_record.get("task_index", -1)))
        subtask_meta = self._get_subtask_meta(task_index, subtask_id)

        phase_record = self._get_required(
            self.phases_by_episode_subtask,
            (episode_index, subtask_id),
            (
                "phase segment record for "
                f"episode_index={episode_index}, subtask_id={subtask_id}"
            ),
        )
        self._check_fps(phase_record, "phase", episode_index)
        phase_segment = self._find_segment(
            phase_record.get("segments"),
            base_index,
            (
                "phase for "
                f"episode_index={episode_index}, subtask_id={subtask_id}, base_index={base_index}"
            ),
        )
        phase_id = str(phase_segment.get("phase_id", "")).strip()
        if not phase_id:
            raise ValueError(
                f"Missing phase_id for episode_index={episode_index}, "
                f"subtask_id={subtask_id}, base_index={base_index}"
            )
        phase_meta = self._get_phase_meta(task_index, subtask_id, phase_id)

        return {
            "subtask": {
                "subtask_id": subtask_id,
                "description": str(subtask_meta.get("concrete_sub_task", "")),
            },
            "phase": {
                "phase_id": phase_id,
                "category": str(phase_meta.get("category", "")),
                "description": str(phase_meta.get("description", "")),
            },
            "grounding": {
                "objects": self._build_grounding_objects(
                    grounding_record=grounding_record,
                    subtask_meta=subtask_meta,
                    subtask_id=subtask_id,
                    base_index=base_index,
                )
            },
        }

    def _load_task_segments(self) -> dict[int, dict[str, Any]]:
        path = self.dataset_path / CUSTOM_TASK_SEGMENT_FILENAME
        records = self._load_json(path)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a list of task segment records")

        task_segments: dict[int, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid task segment record in {path}: {record}")
            task_index = int(record["task_index"])
            task_segments[task_index] = record
        return task_segments

    def _load_jsonl_by_episode(self, filename: str) -> dict[int, dict[str, Any]]:
        path = self.dataset_path / filename
        records: dict[int, dict[str, Any]] = {}
        for record in self._iter_jsonl(path):
            episode_index = int(record["episode_index"])
            records[episode_index] = record
        return records

    def _load_phase_segments(self) -> dict[tuple[int, str], dict[str, Any]]:
        path = self.dataset_path / CUSTOM_PHASE_SEGMENT_FILENAME
        records: dict[tuple[int, str], dict[str, Any]] = {}
        for record in self._iter_jsonl(path):
            episode_index = int(record["episode_index"])
            subtask_id = str(record.get("sub_task_id", "")).strip()
            if not subtask_id:
                raise ValueError(f"Missing sub_task_id in phase record from {path}: {record}")
            records[(episode_index, subtask_id)] = record
        return records

    def _load_json(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Missing custom annotation file: {path}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _iter_jsonl(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing custom annotation file: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object at {path}:{line_no}, got {type(record)}")
                yield record

    def _check_fps(self, record: dict[str, Any], record_name: str, episode_index: int) -> None:
        if self.dataset_fps is None:
            return
        annotation_fps = record.get("original_fps", record.get("fps"))
        if annotation_fps is None:
            return
        if abs(float(annotation_fps) - self.dataset_fps) > 1e-4:
            raise ValueError(
                f"{record_name} annotation fps mismatch for episode_index={episode_index}: "
                f"annotation_fps={annotation_fps}, dataset_fps={self.dataset_fps}"
            )

    def _get_subtask_meta(self, task_index: int, subtask_id: str) -> dict[str, Any]:
        task_record = self._get_required(
            self.task_segments,
            task_index,
            f"task segment metadata for task_index={task_index}",
        )
        task_segment = task_record.get("task_segment")
        if not isinstance(task_segment, dict):
            raise ValueError(f"Missing task_segment for task_index={task_index}")

        for subtask in task_segment.get("sub_tasks", []) or []:
            if str(subtask.get("sub_task_id", "")).strip() == subtask_id:
                if not subtask.get("concrete_sub_task"):
                    raise ValueError(
                        f"Missing concrete_sub_task for task_index={task_index}, "
                        f"subtask_id={subtask_id}"
                    )
                return subtask
        raise ValueError(f"Missing subtask metadata for task_index={task_index}, subtask_id={subtask_id}")

    def _get_phase_meta(self, task_index: int, subtask_id: str, phase_id: str) -> dict[str, Any]:
        subtask_meta = self._get_subtask_meta(task_index, subtask_id)
        for phase in subtask_meta.get("phases", []) or []:
            if str(phase.get("phase_id", "")).strip() == phase_id:
                if not phase.get("category") or not phase.get("description"):
                    raise ValueError(
                        f"Missing phase category/description for task_index={task_index}, "
                        f"subtask_id={subtask_id}, phase_id={phase_id}"
                    )
                return phase
        raise ValueError(
            f"Missing phase metadata for task_index={task_index}, "
            f"subtask_id={subtask_id}, phase_id={phase_id}"
        )

    def _build_grounding_objects(
        self,
        grounding_record: dict[str, Any],
        subtask_meta: dict[str, Any],
        subtask_id: str,
        base_index: int,
    ) -> list[dict[str, Any]]:
        object_manipulation = {
            str(obj.get("id", "")).strip(): obj.get("is_manipulated")
            for obj in subtask_meta.get("objects", []) or []
            if str(obj.get("id", "")).strip()
        }
        if not object_manipulation:
            raise ValueError(f"Missing object metadata for subtask_id={subtask_id}")

        packed_objects = []
        for obj in grounding_record.get("objects", []) or []:
            object_key = str(obj.get("object_key", "")).strip()
            if object_key == "0::0":
                if self.include_gripper:
                    packed_objects.append(self._pack_grounding_object(obj, base_index, None))
                continue

            prefix, sep, object_id = object_key.partition("::")
            if sep != "::" or prefix != subtask_id:
                continue
            if object_id not in object_manipulation:
                raise ValueError(
                    f"Missing is_manipulated metadata for object_key={object_key}, "
                    f"subtask_id={subtask_id}"
                )
            packed_objects.append(
                self._pack_grounding_object(obj, base_index, object_manipulation[object_id])
            )

        expected_count = len(object_manipulation) + (1 if self.include_gripper else 0)
        if len(packed_objects) != expected_count:
            raise ValueError(
                f"Grounding object count mismatch for episode_index={grounding_record.get('episode_index')}, "
                f"subtask_id={subtask_id}, base_index={base_index}: "
                f"expected={expected_count}, found={len(packed_objects)}"
            )
        return packed_objects

    def _pack_grounding_object(
        self,
        obj: dict[str, Any],
        base_index: int,
        is_manipulated: bool | None,
    ) -> dict[str, Any]:
        object_key = str(obj.get("object_key", "")).strip()
        bboxes = obj.get("bbox")
        if not isinstance(bboxes, list):
            raise ValueError(f"Missing bbox list for object_key={object_key}")
        if base_index < 0 or base_index >= len(bboxes):
            raise ValueError(
                f"BBox index out of range for object_key={object_key}: "
                f"base_index={base_index}, bbox_len={len(bboxes)}"
            )
        bbox = bboxes[base_index]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(
                f"Invalid bbox for object_key={object_key}, base_index={base_index}: {bbox}"
            )
        return {
            "object_key": object_key,
            "name": str(obj.get("name", "")),
            "bbox": bbox,
            "is_manipulated": is_manipulated,
        }

    def _find_segment(
        self,
        segments: Any,
        base_index: int,
        description: str,
    ) -> dict[str, Any]:
        if not isinstance(segments, list):
            raise ValueError(f"Missing segments list for {description}")
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            start_idx = int(segment["start_idx"])
            end_idx = int(segment["end_idx"])
            if start_idx <= base_index <= end_idx:
                return segment
        raise ValueError(f"Missing {description}")

    def _get_required(self, mapping: dict, key: Any, description: str) -> Any:
        if key not in mapping:
            raise ValueError(f"Missing {description}")
        return mapping[key]


class CustomLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_cfg = kwargs.get("data_cfg", None)
        custom_annotation_cfg = data_cfg.get("custom_annotation", {}) if data_cfg else {}
        include_gripper = custom_annotation_cfg.get("include_gripper", True)
        self.annotation_index = CustomAnnotationIndex(
            dataset_path=self.dataset_path,
            dataset_fps=self.lerobot_info_meta.get("fps"),
            include_gripper=include_gripper not in ["False", False],
        )
        self._annotation_context: tuple[int, int] | None = None

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        self._annotation_context = (int(trajectory_id), int(base_index))
        return super().get_step_data(trajectory_id, base_index)

    def _pack_sample(self, data: dict) -> dict:
        if self._annotation_context is None:
            raise ValueError("Custom annotation context is missing before packing sample")
        episode_index, base_index = self._annotation_context
        sample = super()._pack_sample(data)
        sample.update(self.annotation_index.get_sample_fields(episode_index, base_index))
        return sample
