import os
import warnings
from dataclasses import dataclass
from typing import Any

from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
import json
import re
from datetime import datetime, timedelta
from enum import Enum, auto

import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.parser import isoparse
from tqdm.autonotebook import tqdm

pd.options.mode.chained_assignment = None

DATA_ROOT = "../data"
SESSION_MERGE_INTERVAL = 10  # minutes


@dataclass
class Engagement:
    start: datetime
    end: datetime
    page: str


@dataclass
class Session:
    engagements: list[Engagement]

    @property
    def n_unique_pages(self):
        return len(set([e.page for e in self.engagements]))

    @property
    def duration(self):
        return (self.engagements[-1].end - self.engagements[0].start).total_seconds()


class QuestionType(Enum):
    END_OF_CHAPTER_REVIEW = auto()
    INLINE = auto()


class Dataset:
    RAW_DATA_PATH = "../data/raw"

    def __init__(
        self, name: str, final_grade_column: str, multi_class: bool = False
    ) -> None:
        """
        name: str, the name of the dataset, also the folder name to load data from
        final_grade_column: str, the name of the column

        The dataset folder should at least contain the following files:
        - page_views.csv
        - responses.csv
        - final_grades.csv
        """
        # Check if all files exist
        for fname in ["page_views.csv", "responses.csv", "final_grades.csv"]:
            fpath = os.path.join(self.RAW_DATA_PATH, name, fname)
            if not os.path.exists(fpath):
                raise ValueError(f"File {fname} not found in {name} folder")

        # Read page views, exclude rows for review questions
        v = pd.read_csv(
            os.path.join(self.RAW_DATA_PATH, name, "page_views.csv"), low_memory=False
        )
        v = v[~v["page"].str.contains("Review Questions", na=False)]
        v = v[v["chapter"].notnull()]

        # Load responses
        r = pd.read_csv(
            os.path.join(self.RAW_DATA_PATH, name, "responses.csv"), low_memory=False
        )

        # Load final grades, exclude students without final grade
        g = pd.read_csv(
            os.path.join(self.RAW_DATA_PATH, name, "final_grades.csv"), low_memory=False
        )
        g = g.rename(columns={final_grade_column: "final_grade"})
        g = g[["student_id", "final_grade"]]
        g.set_index("student_id", inplace=True)
        g = g[g.notna().all(axis=1)]

        # Generate label for final grade
        if multi_class:
            # Multi-class classification
            # 0 if A, A+, A-
            # 1 if B, B+, B-
            # 2 if below B, requires intervention
            g["label"] = g["final_grade"].apply(
                lambda x: (
                    0 if x in ["A", "A+", "A-"] else 1 if x in ["B", "B+", "B-"] else 2
                )
            )
        else:
            # Binary classification
            # Pos: low performance students, needs intervention
            # Neg: high performance students, no need for intervention
            g["label"] = g["final_grade"].apply(
                lambda x: 0 if x in ["A", "A+", "A-"] else 1
            )

        self.responses = r
        self.views = v
        self.grades = g
        self.chapters = self._get_chapter_names()

    def process(self, fillna=True):
        all_features = {}
        for i in tqdm(range(1, len(self.chapters))):
            features_by_student = self._generate_chapter_features_by_student(i)
            for student, features in features_by_student.items():
                merged_features = {**all_features.get(student, {}), **features}
                all_features[student] = merged_features

        features_df = pd.DataFrame.from_dict(all_features, orient="index")

        if fillna:
            features_df = features_df.fillna(0.0)

        # Join features with grades
        n_before_join = len(features_df)
        joined_df = features_df.join(self.grades, how="inner")
        n_after_join = len(joined_df)

        print(f"Total number of students in the dataset: {n_after_join}")
        print(f"Excluded {n_before_join - n_after_join} students without final grade")

        return joined_df.reset_index().rename(columns={"index": "student_id"})

    def _get_chapter_names(self):
        """
        Get the chapter names from the page views dataframe in order
        """
        chapters = [c for c in set(self.views["chapter"]) if "Chapter" in c]
        chapters = sorted(
            chapters,
            key=lambda c: int(
                re.findall(r"\d+", c)[0]
            ),  # extract the first number from chapter name for sorting
        )
        chapters = [""] + chapters  # so that index aligns with actual chapter number

        # Sanity check in case the coursebook version is not expected
        if len(chapters) > 13:
            print("More than 12 chapters found. Chapters after 12 are removed")
            chapters = chapters[:13]
        return chapters

    def _generate_chapter_features_by_student(self, n) -> dict[str, Any]:
        """
        n: chapter number, 1-indexed
        The output is a dict from student id to a dict of features
        """

        # Filter data by chapter name
        chapter = self.chapters[n]
        chapter_views = self.views[self.views["chapter"] == chapter]
        chapter_responses = self.responses[self.responses["chapter"] == chapter]

        # Extract valid reading sessions for each student
        sessions_by_student = get_sessions_by_student(chapter_views)

        # Get session durations by students,
        # in the form of {student_id:[duration1, duration2, ...]}
        durations_by_student = {
            student: [s.duration for s in sessions]
            for student, sessions in sessions_by_student.items()
            if len(sessions) > 0
        }

        # Performance on end of chapter review questions
        eoc_performance_by_student_raw = get_performance_by_student(
            chapter_responses, QuestionType.END_OF_CHAPTER_REVIEW
        )
        eoc_first_attempt_by_student = {
            student: performance["first_performance"]
            for student, performance in eoc_performance_by_student_raw.items()
        }

        # Performance on inline questions
        inline_performance_by_student_raw = get_performance_by_student(
            chapter_responses, QuestionType.INLINE
        )
        inline_first_attempt_by_student = {
            student: performance["first_performance"]
            for student, performance in inline_performance_by_student_raw.items()
        }
        inline_last_attempt_by_student = {
            student: performance["last_performance"]
            for student, performance in inline_performance_by_student_raw.items()
        }
        inline_avg_attempts_by_student = {
            student: performance["avg_attempts"]
            for student, performance in inline_performance_by_student_raw.items()
        }

        # Engagement ratio
        engagement_ratio_by_student = get_engagement_ratio_by_student(chapter_views)

        # End of chapter summary length (in tokens)
        summary_len_by_student = get_chapter_summary_len(chapter_responses)

        # Average number of pages per session
        mean_session_pages_by_student = {
            student: np.mean([s.n_unique_pages for s in sessions])
            for student, sessions in sessions_by_student.items()
            if len(sessions) > 0
        }

        # 2.3 build features
        ret = {
            s: {
                # Number of reading session in this chapter
                f"ch{n:02d}_n": float(len(durations)),
                # Average session duration in minutes
                f"ch{n:02d}_d_mean": np.mean(durations) / 60,
                # Max session duration in minutes
                f"ch{n:02d}_d_max": np.max(durations) / 60,
                # Total session durations in minutes
                f"ch{n:02d}_d_total": np.sum(durations) / 60,
                # Student performance on end of chapter question points (percentage)
                f"ch{n:02d}_q": eoc_first_attempt_by_student.get(s),
                # Student engagement ratio
                f"ch{n:02d}_er": engagement_ratio_by_student.get(s),
                # Student points ratio on embedded questions
                f"ch{n:02d}_pr_first": inline_first_attempt_by_student.get(s),
                f"ch{n:02d}_pr_last": inline_last_attempt_by_student.get(s),
                f"ch{n:02d}_attempts_mean": inline_avg_attempts_by_student.get(s),
                # Length of summary
                f"ch{n:02d}_summary_len": summary_len_by_student.get(s),
                # Average number of pages per session
                f"ch{n:02d}_sp_mean": mean_session_pages_by_student.get(s),
            }
            for s, durations in durations_by_student.items()
        }

        return ret


def merge_engagements(engagements: list[Engagement], interval=10) -> list[Session]:
    """
    Group engagements together into sections.
    We group segments if the starting time of the next one
    is less than `interval` minutes late than the previous end time.

    Returns:
    list of groups of segments
    """
    engagements = sorted(engagements, key=lambda e: (e.start, e.end))

    # Set up two pointers to mark the start and end of the current group
    l, r = 0, 0
    ret: list[Session] = []

    while l < len(engagements):
        r = l

        # We use max_end to mark the end time of the current group
        max_end = engagements[l].end
        group = []

        # Keep moving r to the next segment
        # while the current segment can can be merged into the current group
        while r < len(engagements) and (
            engagements[r].start - max_end <= timedelta(minutes=interval)
        ):
            group.append(engagements[r])
            max_end = max(max_end, engagements[r].end)
            r += 1

        # Append the current group to result
        ret.append(Session(engagements=group))

        # Let l point to the next segment after r
        l = r

    return ret


def get_sessions_by_student(chapter_views: pd.DataFrame) -> dict[str, list[Session]]:
    all_sessions = {}
    student_ids = set(list(chapter_views["student_id"]))
    for student in student_ids:
        student_views = chapter_views[chapter_views["student_id"] == student]
        sessions = get_sessions(student_views)

        all_sessions[student] = sessions
    return all_sessions


def get_performance_by_student(
    chapter_responses: pd.DataFrame, question_type: QuestionType
) -> dict[str, dict[str, float]]:
    """
    Get student's performance on end of chapter questions or inline questions.
    """
    if question_type == QuestionType.END_OF_CHAPTER_REVIEW:
        chapter_questions = chapter_responses[
            chapter_responses["page"].str.endswith("Review Questions")
        ]
    else:
        chapter_questions = chapter_responses[
            ~chapter_responses["page"].str.endswith("Review Questions")
        ]

    # Get the total points possible for the chapter
    # Some questions may not have points (e.g., free response) which will be represented as NaN, they will be ignored
    # Some questions allow multiple attempts,
    # Note: UCLA and UCSD are using different versions of the coursebook, so we use prompt instead of item id to identify unique questions
    total_points_possible = chapter_questions.drop_duplicates(subset=["prompt"])[
        "points_possible"
    ].sum()

    # Get the performance of 1st attempts
    first_attempts = chapter_questions[chapter_questions["attempt"] == 1]

    first_performance_by_student = (
        first_attempts.groupby("student_id")
        .apply(
            lambda x: (
                x["points_earned"].sum() / total_points_possible
                if x["points_possible"].sum()
                > 0  # Some chapter may not have any questions
                else -1.0
            )
        )
        .to_dict()
    )

    # End of chapter review questions only allow 1 attempt
    if question_type == QuestionType.END_OF_CHAPTER_REVIEW:
        return {
            student: {"first_performance": performance}
            for student, performance in first_performance_by_student.items()
        }

    # Last attempts
    last_attempts = chapter_questions.loc[
        chapter_questions.groupby(["student_id", "item_id"])["attempt"].idxmax()
    ]

    # Get average number of attempts per student
    avg_attempts_by_student = last_attempts.groupby("student_id")["attempt"].mean()

    # Get the performance of last attempts
    last_performance_by_student = (
        last_attempts.groupby("student_id")
        .apply(
            lambda x: (
                x["points_earned"].sum() / total_points_possible
                if x["points_possible"].sum() > 0
                else -1.0
            )
        )
        .to_dict()
    )

    result_dict = {
        student: {
            "first_performance": first_performance_by_student[student],
            "last_performance": last_performance_by_student[student],
            "avg_attempts": avg_attempts_by_student.get(student, 0),
        }
        for student in first_performance_by_student.keys()
    }

    return result_dict


def get_sessions(student_views):
    if len(student_views) == 0:
        return []

    # Sort by record time
    student_views = student_views.sort_values("dt_accessed")

    # get rows with not empty trace and parse the trace into JSON
    student_views = student_views.dropna(subset=["trace"])
    student_views["trace"] = student_views["trace"].apply(lambda x: json.loads(x))
    student_views = student_views[["page", "trace"]]

    # explode by trace so that each row represent a single trace
    # a trace is like: {'timestamp': '2022-03-30T00:43:14.232Z', 'switched_to': 'engaged'}
    student_views = student_views.explode("trace")
    student_views = student_views.reset_index().drop(columns=["index"])

    # convert df into list of tuple of (page_name, trace)
    student_record = list(student_views.to_records(index=False))

    # create a list of (start, end, page) to represent engagement segments
    engagements: list[Engagement] = []
    for i in range(len(student_record) - 1):
        current_page, current_trace = student_record[i]
        _, next_trace = student_record[i + 1]

        if ("engaged" in current_trace["switched_to"]) and (
            "engaged" not in next_trace["switched_to"]
        ):
            start = isoparse(current_trace["timestamp"]).replace(tzinfo=None)
            end = isoparse(next_trace["timestamp"]).replace(tzinfo=None)

            # Skip invalid records
            if start > end:
                continue

            engagements.append(Engagement(start, end, current_page))

    # merge engagement spans into sessions
    sessions = merge_engagements(engagements, SESSION_MERGE_INTERVAL)

    # exclude sessions with less than 60s duration
    sessions = [s for s in sessions if s.duration >= 60]

    return sessions


def get_engagement_ratio_by_student(chapter_views):
    """
    Get student engagement ratio given a view dataframe with a specific chapter
    """
    df = (
        chapter_views.groupby("student_id")
        .agg(
            engaged_sum=("engaged", "sum"),
            idle_sum=("idle_brief", "sum"),
            off_page_sum=("off_page_brief", "sum"),
        )
        .fillna(0.0)
    ).reset_index()

    # Compute ratio
    df["ratio"] = df["engaged_sum"] / (
        df["engaged_sum"] + df["idle_sum"] + df["off_page_sum"]
    )

    return dict(df[["student_id", "ratio"]].to_records(index=False))


def get_chapter_summary_len(chapter_responses: pd.DataFrame):
    prompt = "In a paragraph, summarize the main idea(s) in this chapter."
    df = chapter_responses.dropna(subset=["prompt"])
    df = df[df["prompt"] == prompt]
    df = df.dropna(subset=["response"])
    df["summary_len"] = df["response"].str.split().apply(len)
    return dict(df[["student_id", "summary_len"]].to_records(index=False))
