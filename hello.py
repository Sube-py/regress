# type: ignore
import asyncio
import re
import pandas as pd
from lifelines.fitters.coxph_fitter import CoxPHFitter as Cox
import matplotlib.pyplot as plt
import numpy as np
from lifelines import utils
from lifelines.utils.printer import Printer
from pathlib import Path

current_path = Path(__file__).parent

class CoxPHFitter(Cox):
    def print_summary(self, decimals=2, style=None, columns=None, **kwargs):
        justify = utils.string_rjustify(25)

        headers = []

        if utils.CensoringType.is_interval_censoring(self):
            headers.append(("lower bound col", "'%s'" % self.lower_bound_col))
            headers.append(("upper bound col", "'%s'" % self.upper_bound_col))
        else:
            headers.append(("duration col", "'%s'" % self.duration_col))

        if self.event_col:
            headers.append(("event col", "'%s'" % self.event_col))
        if self.weights_col:
            headers.append(("weights col", "'%s'" % self.weights_col))
        if self.entry_col:
            headers.append(("entry col", "'%s'" % self.entry_col))
        if self.cluster_col:
            headers.append(("cluster col", "'%s'" % self.cluster_col))
        if isinstance(self.penalizer, np.ndarray) or self.penalizer > 0:
            headers.append(("penalizer", self.penalizer))
            headers.append(("l1 ratio", self.l1_ratio))
        if self.robust or self.cluster_col:
            headers.append(("robust variance", True))
        if self.strata:
            headers.append(("strata", self.strata))
        if self.baseline_estimation_method == "spline":
            headers.append(("number of baseline knots", self.n_baseline_knots))
        if self.baseline_estimation_method == "piecewise":
            headers.append(("location of breaks", self.breakpoints))

        headers.extend(
            [
                ("baseline estimation", self.baseline_estimation_method),
                ("number of observations", "{:g}".format(self.weights.sum())),
                (
                    "number of events observed",
                    "{:g}".format(self.weights[self.event_observed > 0].sum()),
                ),
                (
                    "partial log-likelihood" if self.baseline_estimation_method == "breslow" else "log-likelihood",
                    "{:.{prec}f}".format(self.log_likelihood_, prec=decimals),
                ),
                ("time fit was run", self._time_fit_was_called),
            ]
        )

        footers = []
        sr = self.log_likelihood_ratio_test()

        if self.baseline_estimation_method == "breslow":
            footers.extend(
                [
                    (
                        "Concordance",
                        "{:.{prec}f}".format(self.concordance_index_, prec=decimals),
                    ),
                    (
                        "Partial AIC",
                        "{:.{prec}f}".format(self.AIC_partial_, prec=decimals),
                    ),
                ]
            )
        elif self.baseline_estimation_method in ["spline", "piecewise"]:
            footers.append(("AIC", "{:.{prec}f}".format(self.AIC_, prec=decimals)))

        footers.extend(
            [
                (
                    "log-likelihood ratio test",
                    "{:.{prec}f} on {} df".format(sr.test_statistic, sr.degrees_freedom, prec=decimals),
                ),
                (
                    "-log2(p) of ll-ratio test",
                    "{:.{prec}f}".format(-utils.quiet_log2(sr.p_value), prec=decimals),
                ),
            ]
        )

        p = Printer(self, headers, footers, justify, kwargs, decimals, columns)
        path = kwargs.pop("save_path", None)
        html = p.to_html()

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html)
        else:
            print(html)


class COX:
    @classmethod
    def categorize_outcome(cls, outcome):
        if outcome == "Suicide and Self-Inflicted Injury":
            return 1
        elif outcome == "Alive":
            # return "alive"
            return 0
        else:
            return 0

    @classmethod
    def cox_by_age_group(cls, data_frame: pd.DataFrame) -> None:
        cph = CoxPHFitter()
        df = data_frame
        df["age_start"] = df["Age recode with <1 year olds"].str.extract(r"(\d+)")[0].astype(int)
        labels = ["0-19", "20-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        df["age_group"] = pd.cut(
            df["age_start"],
            bins=[0, 20, 40, 50, 60, 70, 80, 100],
            labels=labels,
            right=False,
        )

        cph.fit(
            df,
            duration_col="Survival months",
            event_col="event",
            formula="age_group",
            show_progress=False,
        )
        cph.print_summary(style="html")

    @classmethod
    def cox_by_gender(cls, data_frame: pd.DataFrame) -> None:
        cph = CoxPHFitter()
        df = data_frame
        cph.fit(
            df,
            duration_col="Survival months",
            event_col="event",
            formula="Sex",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_gender.html")

    @classmethod
    def cox_by_race(cls, data_frame: pd.DataFrame) -> None:
        df = data_frame
        race_map = {
            "White": "W",
            "Black": "B",
            # "Asian or Pacific Islander": "API",
        }

        df["race"] = df["Race recode"].apply(lambda x: race_map.get(x, "Other"))
        cph = CoxPHFitter()
        new_df = cls.set_reference(df, "race", ["W", "B", "Other"])
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_race.html")

    @classmethod
    def cox_by_nhia(cls, data_frame: pd.DataFrame):
        cph = CoxPHFitter()
        df = data_frame
        df["NHIA"] = df["Origin recode NHIA"].apply(lambda x: "Non-Hisp" if "Non" in x else "Hispanic")
        cph.fit(
            df,
            duration_col="Survival months",
            event_col="event",
            formula="NHIA",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_nhia.html")

    @classmethod
    def set_reference(
        cls,
        data_frame: pd.DataFrame,
        column: str,
        categories: list[str],
        return_dummie_only: bool = False,
    ):
        data_frame["reference"] = pd.Categorical(
            data_frame[column],
            categories=categories,
            ordered=True,
        )
        df_dum = pd.get_dummies(data_frame, columns=["reference"], drop_first=True)
        if return_dummie_only:
            return df_dum
        columns_to_use = ["Survival months", "event"] + [col for col in df_dum.columns if col.startswith("reference_")]
        return df_dum[columns_to_use]

    @classmethod
    def cox_by_marital_status(cls, data_frame: pd.DataFrame):
        cph = CoxPHFitter()

        def outcome(x):
            if "Married" in x:
                return "Married"

            if "Divorced" in x or "Separated" in x or "Widowed" in x:
                return "DSW"

            return "Single"

        df = data_frame
        df["Marital"] = df["Marital status at diagnosis"].apply(outcome)
        new_df = cls.set_reference(df, "Marital", ["Married", "DSW", "Single"])
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_marital_status.html")

    @classmethod
    def cox_by_surgery(cls, data_frame: pd.DataFrame):
        cph = CoxPHFitter()

        df = data_frame
        df["surgery"] = df["Reason no cancer-directed surgery"].apply(
            lambda x: "Performed" if x == "Surgery performed" else "No"
        )
        new_df = cls.set_reference(df, "surgery", ["Performed", "No"])
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_surgery.html")

    @classmethod
    def cox_by_stage(cls, data_frame: pd.DataFrame):
        cph = CoxPHFitter()
        df = data_frame
        new_df = cls.set_reference(df, "Stage", ["In situ", "Localized", "Regional", "Distant"])
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_stage.html")

    @classmethod
    def cox_by_grade(cls, data_frame: pd.DataFrame):
        cph = CoxPHFitter()
        df = data_frame
        grade_map = {
            "1": "I",
            "2": "II",
            "3": "III",
            "I": "I",
            "II": "II",
            "III": "III",
        }

        def outcome(row: pd.Series):
            for col in ["8th", "7th", "6th"]:
                r = str(row[col])
                if "Blank" not in r:
                    return grade_map.get(r, "IV")
            raise ValueError("No grade found")

        df["grade"] = df.apply(outcome, axis=1)
        cph.fit(
            df,
            duration_col="Survival months",
            event_col="event",
            formula="grade",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_grade.html")

    @classmethod
    def cox_by_year(cls, data_frame: pd.DataFrame):
        cph = CoxPHFitter()
        df = data_frame
        new_df = cls.set_reference(df, "Year of diagnosis", list(range(2004, 2020)))
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_year.html")

    @classmethod
    def cox_by_radio(cls, data_frame: pd.DataFrame):
        cph = CoxPHFitter()

        df = data_frame
        df["surgery"] = df["RX Summ--Surg/Rad Seq"].apply(
            lambda x: "Performed" if x == "Radiation after surgery" else "No/Unknown/Refused"
        )
        new_df = cls.set_reference(
            df,
            "surgery",
            [
                "No/Unknown/Refused",
                "Performed",
            ],
        )
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_radio.html")

    @classmethod
    def cox_by_income(cls, data_frame: pd.DataFrame):
        cph = CoxPHFitter()

        def extract_min_income(income_range):
            # 使用正则表达式提取第一个数字
            match = re.search(r"\$(\d{1,3}(?:,\d{3})*)", income_range)
            if match:
                return int(match.group(1).replace(",", ""))
            raise ValueError(f"Invalid income range: {income_range}")

        df = data_frame
        df["income"] = df["income"].apply(extract_min_income)
        labels = ["[0, 45000)", "[45000, 75000)", "75000+"]
        df["income_group"] = pd.cut(
            df["income"],
            bins=[0, 45000, 75000, 100000000],
            labels=labels,
            right=False,
        )
        # new_df = cls.set_reference(
        #     df,
        #     "Year",
        #     list(range(2004, 2020)),
        # )
        cph.fit(
            df,
            duration_col="Survival months",
            event_col="event",
            formula="income_group",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_income.html")

    @classmethod
    def cox_by_chemo(cls, data_frame: pd.DataFrame):
        cph = CoxPHFitter()

        df = data_frame
        df["chemo"] = df["Chemotherapy recode"].apply(lambda x: "Performed" if x == "Yes" else "No/Unknown")
        new_df = cls.set_reference(
            df,
            "chemo",
            [
                "No/Unknown",
                "Performed",
            ],
        )
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_chemo.html")

    @classmethod
    def multi_cox(cls):
        parent_dir = Path(__file__).parent
        x = parent_dir / "data.xlsx"
        df = pd.read_excel(x)

        def marital_outcome(x):
            if "Married" in x:
                return "Married"

            if "Divorced" in x or "Separated" in x or "Widowed" in x:
                return "DSW"

            return "Single"

        df["Marital"] = df["Marital status at diagnosis"].apply(marital_outcome)
        df["new_mari"] = pd.Categorical(
            df["Marital"],
            categories=["Married", "DSW", "Single"],
            ordered=True,
        )
        df = pd.get_dummies(df, columns=["new_mari"], drop_first=True)

        # race
        race_map = {
            "White": "W",
            "Black": "B",
            # "Asian or Pacific Islander": "API",
        }

        df["Race"] = df["Race recode"].apply(lambda x: race_map.get(x, "Other"))
        df["new_race"] = pd.Categorical(
            df["Race"],
            categories=["W", "B", "Other"],
            ordered=True,
        )
        df = pd.get_dummies(df, columns=["new_race"], drop_first=True)

        # grade
        grade_map = {
            "1": 1,
            "2": 2,
            "3": 3,
            "I": 1,
            "II": 2,
            "III": 3,
        }

        def grade_outcome(row: pd.Series):
            for col in ["8th", "7th", "6th"]:
                r = str(row[col])
                if "Blank" not in r:
                    return grade_map.get(r, 4)
            raise ValueError("No grade found")

        df["grade"] = df.apply(grade_outcome, axis=1)
        df["new_grade"] = pd.Categorical(
            df["grade"],
            categories=[1, 2, 3, 4],
            ordered=True,
        )
        df = pd.get_dummies(df, columns=["new_grade"], drop_first=True)

        # year
        df["new_year"] = pd.Categorical(
            df["Year of diagnosis"],
            categories=list(range(2004, 2020)),
            ordered=True,
        )
        df = pd.get_dummies(df, columns=["new_year"], drop_first=True)

        def age_outcome(age: int):
            if age < 20:
                return 2
            if age < 40:
                return 4
            if age < 50:
                return 5
            if age < 60:
                return 6
            if age < 70:
                return 7
            return 8

        # age
        df["age_start"] = df["Age recode with <1 year olds"].str.extract(r"(\d+)")[0].astype(int).apply(age_outcome)
        ages = {item for item in df["age_start"]}
        # labels = ["0-19", "20-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        # labels = [0, 2, 4, 5, 6, 7, 8]
        # df["age_group"] = pd.cut(
        #     df["age_start"],
        #     bins=[0, 20, 40, 50, 60, 70, 80, 100],
        #     labels=labels,
        #     right=False,
        # )
        df["new_age"] = pd.Categorical(
            df["age_start"],
            categories=sorted(list(ages)),
            ordered=True,
        )
        df = pd.get_dummies(df, columns=["new_age"], drop_first=True)

        df["gender"] = df["Sex"].apply(lambda x: 1 if x == "Male" else 0)
        df["new_gender"] = pd.Categorical(
            df["gender"],
            categories=[0, 1],
            ordered=True,
        )
        df = pd.get_dummies(df, columns=["new_gender"], drop_first=True)

        used_col = ["Survival months", "event"]
        used_col.extend(col for col in df.columns if col.startswith("new_"))
        print(used_col)
        cph = CoxPHFitter()
        new_df = df[used_col]
        new_df.to_csv("./parsed_data.csv", index=False)
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            show_progress=True,
        )
        cph.print_summary(save_path="./data/multi_cox.html")

        coefficients = cph.params_
        print(coefficients)
        fig, ax = plt.subplots(figsize=(10, 6))

        variables = coefficients.index
        x_pos = np.arange(len(variables))

        ax.barh(x_pos, coefficients, align="center")
        ax.set_yticks(x_pos)
        ax.set_yticklabels(variables)
        ax.set_xlabel("Coefficient")
        ax.set_title("Cox Proportional Hazards Model")
        for i, v in enumerate(coefficients):
            ax.text(v, i, f"{v:.2f}", va="center", ha="right" if v < 0 else "left")
        plt.tight_layout()
        plt.savefig("./data/multi_cox.png")

    @classmethod
    def cox_by_residence(cls, data_frame: pd.DataFrame):
        "Nonmetropolitan"
        cph = CoxPHFitter()

        df = data_frame
        df["residence"] = df["Residence"].apply(
            lambda x: "Non-metropolitan" if "Nonmetropolitan" in x else "Metropolitan"
        )
        new_df = cls.set_reference(
            df,
            "residence",
            ["Metropolitan", "Non-metropolitan"],
        )
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_residence.html")

    @classmethod
    async def main(cls):
        # df = pd.read_excel("./table2.xlsx")
        # df = df[df["COD to site recode"] != "Alive"]
        # df["event"] = df["COD to site recode"].apply(cls.categorize_outcome)
        # df.to_excel("./data.xlsx", index=False)
        df = pd.read_excel(current_path / "data/data.xlsx")
        # cls.cox_by_age_group(df)
        # cls.cox_by_gender(df)
        # cls.cox_by_race(df)
        # cls.cox_by_nhia(df)
        cls.cox_by_marital_status(df)
        # cls.cox_by_surgery(df)
        # cls.cox_by_stage(df)
        # cls.cox_by_grade(df)
        # cls.cox_by_year(df)
        # cls.cox_by_radio(df)
        # cls.cox_by_chemo(df)
        # cls.cox_by_income(df)
        # cls.multi_cox()
        # cls.cox_by_residence(df)

    @classmethod
    def run(cls):
        asyncio.new_event_loop().run_until_complete(cls.main())


if __name__ == "__main__":
    COX.run()
