from pathlib import Path
import pandas as pd


def load_student_enquiries(file_path):
    columns = [
        "enquiry_id", "course_id", "student_name", "email",
        "phone", "quoted_fee", "discount_pct"
    ]
    df = pd.read_csv(file_path)
    return df[columns]


def clean_enquiry_data(df):
    df = df.dropna(subset=["email"])
    df = df.drop_duplicates(subset=["enquiry_id"], keep="first")
    return df.copy()


def merge_with_courses(enquiries_df, ref_file_path):
    courses = pd.read_csv(ref_file_path)
    return enquiries_df.merge(courses, on="course_id", how="inner")


def add_net_quoted_fee(df):
    result = df.copy()
    result["net_quoted_fee"] = (
        result["quoted_fee"] * (1 - result["discount_pct"] / 100)
    ).round(2)
    return result


def aggregate_by_course(df):
    result = (
        df.groupby("course_name", as_index=False)
        .agg(
            avg_net_quoted_fee=("net_quoted_fee", "mean"),
            enquiry_count=("enquiry_id", "count"),
        )
    )
    return result.reset_index(drop=True)


def finalize_summary(df):
    result = df[
        ["course_name", "avg_net_quoted_fee", "enquiry_count"]
    ].copy()
    result["avg_net_quoted_fee"] = result["avg_net_quoted_fee"].round(2)
    return result.sort_values("course_name").reset_index(drop=True)


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"

    enquiries = load_student_enquiries(data_dir / "training_enquiries.csv")
    cleaned = clean_enquiry_data(enquiries)
    merged = merge_with_courses(cleaned, data_dir / "training_courses.csv")
    fees_added = add_net_quoted_fee(merged)
    summary = aggregate_by_course(fees_added)
    final_summary = finalize_summary(summary)

    print(final_summary.to_string(index=False))
