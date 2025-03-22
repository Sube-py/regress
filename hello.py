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
                    "partial log-likelihood"
                    if self.baseline_estimation_method == "breslow"
                    else "log-likelihood",
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
                    "{:.{prec}f} on {} df".format(
                        sr.test_statistic, sr.degrees_freedom, prec=decimals
                    ),
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
        df["age_start"] = df["Age"].str.extract(r"(\d+)")[0].astype(int)
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
        cph.print_summary(save_path=current_path / "data/cox_by_age_group.html")

    @classmethod
    def cox_by_age(cls, data_frame: pd.DataFrame) -> None:
        cph = CoxPHFitter()
        df = data_frame
        df["age_start"] = df["Age"].str.extract(r"(\d+)")[0].astype(int)
        df["age"] = df["age_start"].apply(lambda x: x >= 55)

        cph.fit(
            df,
            duration_col="Survival months",
            event_col="event",
            formula="age",
            show_progress=False,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_age_55.html")

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
        df["NHIA"] = df["Origin recode NHIA"].apply(
            lambda x: "Non-Hisp" if "Non" in x else "Hispanic"
        )
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
        columns_to_use = ["Survival months", "event"] + [
            col for col in df_dum.columns if col.startswith("reference_")
        ]
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
        df["surgery"] = df["Surgery"].apply(
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
        new_df = cls.set_reference(df, "Stage", ["Localized", "Regional", "Distant"])
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
        # grade_map = {
        #     "1": "I",
        #     "2": "II",
        #     "3": "III",
        #     "I": "I",
        #     "II": "II",
        #     "III": "III",
        # }

        # def outcome(row: pd.Series):
        #     for col in ["8th", "7th", "6th"]:
        #         r = str(row[col])
        #         if "Blank" not in r:
        #             return grade_map.get(r, "IV")
        #     raise ValueError("No grade found")
        new_df = cls.set_reference(df, "Grade", [1, 2, 3, 4])

        # df["grade"] = df.apply(outcome, axis=1)
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            # formula="Grade",
            show_progress=True,
        )
        cph.print_summary(save_path=current_path / "data/cox_by_grade.html")

    @classmethod
    def cox_by_year(cls, data_frame: pd.DataFrame):
        cph = CoxPHFitter()
        df = data_frame
        new_df = cls.set_reference(df, "Year", list(range(2004, 2020)))
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
            lambda x: "Performed"
            if x == "Radiation after surgery"
            else "No/Unknown/Refused"
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
        df["chemo"] = df["Chemotherapy recode"].apply(
            lambda x: "Performed" if x == "Yes" else "No/Unknown"
        )
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
        x = parent_dir / "data" / "data.xlsx"
        df = pd.read_excel(x)

        def marital_outcome(x):
            if "Married" in x:
                return "Married"

            if "Divorced" in x or "Separated" in x or "Widowed" in x:
                return "DSW"

            return "Single"

        # df["Marital"] = df["Marital status at diagnosis"].apply(marital_outcome)
        # df["new_mari"] = pd.Categorical(
        #     df["Marital"],
        #     categories=["Married", "DSW", "Single"],
        #     ordered=True,
        # )
        # df = pd.get_dummies(df, columns=["new_mari"], drop_first=True)

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
        # grade_map = {
        #     "1": 1,
        #     "2": 2,
        #     "3": 3,
        #     "I": 1,
        #     "II": 2,
        #     "III": 3,
        # }

        # def grade_outcome(row: pd.Series):
        #     for col in ["8th", "7th", "6th"]:
        #         r = str(row[col])
        #         if "Blank" not in r:
        #             return grade_map.get(r, 4)
        #     raise ValueError("No grade found")

        # df["grade"] = df.apply(grade_outcome, axis=1)
        df["new_grade"] = pd.Categorical(
            df["Grade"],
            categories=[3, 1, 2, 4],
            ordered=True,
        )
        df = pd.get_dummies(df, columns=["new_grade"], drop_first=True)

        # year
        # df["new_year"] = pd.Categorical(
        #     df["Year of diagnosis"],
        #     categories=list(range(2004, 2020)),
        #     ordered=True,
        # )
        # df = pd.get_dummies(df, columns=["new_year"], drop_first=True)

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
            if age < 80:
                return 8
            return 9

        def age_55(age: int):
            return int(age >= 55)

        # age
        df["age_start"] = df["Age"].str.extract(r"(\d+)")[0].astype(int).apply(age_55)
        ages = {item for item in df["age_start"]}
        # labels = ["0-19", "20-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        # labels = [0, 2, 4, 5, 6, 7, 8]
        # df["age_group"] = pd.cut(
        #     df["age_start"],
        #     bins=[0, 20, 40, 50, 60, 70, 80, 100],
        #     labels=labels,
        #     right=False,
        # )
        # ages = list(ages)
        # ages.sort()
        # ages.remove(9)
        # ages.insert(0, 9)
        # print(ages)
        df["new_age"] = pd.Categorical(
            df["age_start"],
            categories=[0, 1],
            ordered=True,
        )
        df = pd.get_dummies(df, columns=["new_age"], drop_first=True)

        # def stage_outcome(age: str):
        #     if age == "In situ":
        #         return 0
        #     if age == "Localized":
        #         return 1
        #     if age == "Regional":
        #         return 2
        #     # if age == "Distant":
        #     return 3

        # df["stage_1"] = df["Stage"].apply(stage_outcome)
        # print({item for item in df["stage_1"]})
        df["new_stage"] = pd.Categorical(
            df["Stage"],
            categories=["Localized", "Regional", "Distant"],
            ordered=True,
        )
        df = pd.get_dummies(df, columns=["new_stage"], drop_first=True)

        df["gender"] = df["Sex"].apply(lambda x: 1 if x == "Male" else 0)
        df["new_gender"] = pd.Categorical(
            df["gender"],
            categories=[0, 1],
            ordered=True,
        )
        df = pd.get_dummies(df, columns=["new_gender"], drop_first=True)

        used_col = ["Survival months", "event"]
        cols = [col for col in df.columns if col.startswith("new_")]
        used_col.extend(cols)
        print(used_col)
        cph = CoxPHFitter()
        new_df = df[used_col]
        # new_df.to_csv("./parsed_data.csv", index=False)
        # return
        cph.fit(
            new_df,
            duration_col="Survival months",
            event_col="event",
            show_progress=True,
        )
        cph.print_summary(save_path="./data/multi_cox_no_marital_and_age_55.html")
        return
        coefficients = cph.params_
        confidence_intervals = cph.confidence_intervals_

        # 找到最大和最小的系数值，用于统一坐标轴范围
        all_values = np.concatenate(
            (confidence_intervals.values.flatten(), coefficients.values)
        )
        min_val, max_val = all_values.min(), all_values.max()

        # 创建图形
        fig, axes = plt.subplots(
            len(coefficients), 1, figsize=(10, len(coefficients)), sharex=True
        )

        # 遍历每个特征，绘制误差棒
        for i, (var, coef) in enumerate(coefficients.items()):
            ci_lower, ci_upper = confidence_intervals.loc[var]

            # 计算误差
            xerr = np.array([[coef - ci_lower], [ci_upper - coef]]).reshape(2, 1)

            # 绘制误差棒
            axes[i].errorbar(coef, 0, xerr=xerr, fmt="o", color="b", capsize=5)

            # 显示因素名称
            axes[i].text(
                min_val - 1, 0, var, verticalalignment="center", fontsize=10, ha="right"
            )

            # 设置轴
            axes[i].set_xlim(min_val - 1, max_val + 1)
            axes[i].set_yticks([])
            axes[i].set_xticks([])
            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
            axes[i].spines["left"].set_visible(False)
            axes[i].grid(True, axis="x", linestyle="--", alpha=0.7)

        # 设置总标题
        fig.suptitle("Nomogram with Error Bars", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.show()
        plt.savefig("./multi_cox.png")

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
        # df = df.drop(columns=["COD to site recode"])
        # df.to_excel("./data.xlsx", index=False)
        # df = pd.read_excel(current_path / "data/data.xlsx")
        # cls.cox_by_age_group(df)
        # cls.cox_by_age(df)
        # cls.cox_by_gender(df)
        # cls.cox_by_race(df)
        # cls.cox_by_nhia(df)
        # cls.cox_by_marital_status(df)
        # cls.cox_by_surgery(df)
        # cls.cox_by_stage(df)
        # cls.cox_by_grade(df)
        # cls.cox_by_year(df)
        # # cls.cox_by_radio(df)
        # cls.cox_by_chemo(df)
        # cls.cox_by_income(df)
        # cls.cox_by_residence(df)

        cls.multi_cox()

    @classmethod
    def run(cls):
        asyncio.new_event_loop().run_until_complete(cls.main())


if __name__ == "__main__":
    COX.run()
