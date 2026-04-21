import os
import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model


st.set_page_config(
	page_title="Salary Decider",
	page_icon="chart",
	layout="wide",
)


def app_paths():
	root = os.path.abspath(os.path.dirname(__file__))
	model_file = os.path.join(root, "artifacts", "pipeline.pkl")
	data_file = os.path.join(root, "data", "DataScience_Salaries.csv")
	return root, model_file, data_file


@st.cache_resource
def get_model(model_file: str):
	model_name = os.path.splitext(model_file)[0]
	return load_model(model_name)


@st.cache_data
def get_reference_data(data_file: str):
	return pd.read_csv(data_file)


OCEAN_THEME = {
	"app_bg": "radial-gradient(circle at 10% 0%, #f3f9ff 0%, #eef5ff 35%, #ffffff 100%)",
	"hero_border": "#dbeafe",
	"hero_title": "#0f172a",
	"hero_subtitle": "#334155",
	"hero_chip_bg": "#ecfeff",
	"hero_chip_text": "#0c4a6e",
	"card_bg": "#f8fbff",
	"card_border": "#dbeafe",
	"label_color": "#1d4ed8",
	"value_color": "#0f172a",
}


def inject_styles(theme_tokens):
	st.markdown(
		f"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

		html, body, [class*="css"] {{
			font-family: 'Space Grotesk', sans-serif;
		}}

		.stApp {{
			background: {theme_tokens["app_bg"]};
		}}

		.hero {{
			padding: 1rem 1.1rem;
			border-radius: 16px;
			background: rgba(255, 255, 255, 0.85);
			backdrop-filter: blur(3px);
			border: 1px solid {theme_tokens["hero_border"]};
			box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
			margin-bottom: 0.8rem;
		}}

		.hero-header {{
			display: flex;
			align-items: center;
			justify-content: space-between;
			gap: 0.65rem;
		}}

		.hero h1 {{
			margin: 0;
			font-weight: 700;
			letter-spacing: 0.15px;
			font-size: 1.7rem;
			color: {theme_tokens["hero_title"]};
		}}

		.hero p {{
			margin-top: 0.45rem;
			margin-bottom: 0;
			color: {theme_tokens["hero_subtitle"]};
			font-size: 0.98rem;
			line-height: 1.45;
		}}

		.hero-chip {{
			display: inline-block;
			padding: 0.32rem 0.58rem;
			border-radius: 999px;
			font-size: 0.75rem;
			font-weight: 600;
			background: {theme_tokens["hero_chip_bg"]};
			color: {theme_tokens["hero_chip_text"]};
			border: 1px solid #bae6fd;
		}}

		.hero-grid {{
			display: grid;
			grid-template-columns: repeat(3, minmax(0, 1fr));
			gap: 0.55rem;
			margin-top: 0.8rem;
		}}

		.hero-mini {{
			border: 1px solid #e2e8f0;
			border-radius: 12px;
			padding: 0.5rem 0.6rem;
			background: #ffffff;
		}}

		.hero-mini-label {{
			font-size: 0.7rem;
			text-transform: uppercase;
			letter-spacing: 0.08em;
			color: #64748b;
		}}

		.hero-mini-value {{
			margin-top: 0.2rem;
			font-size: 0.9rem;
			font-weight: 600;
			color: #0f172a;
		}}

		@media (max-width: 900px) {{
			.hero-header {{
				flex-direction: column;
				align-items: flex-start;
			}}

			.hero-grid {{
				grid-template-columns: 1fr;
			}}
		}}

		.result-card {{
			border-radius: 14px;
			padding: 1rem 1.1rem;
			border: 1px solid {theme_tokens["card_border"]};
			background: {theme_tokens["card_bg"]};
		}}

		.result-label {{
			font-size: 0.82rem;
			color: {theme_tokens["label_color"]};
			text-transform: uppercase;
			letter-spacing: 0.08em;
		}}

		.result-value {{
			font-size: 2rem;
			font-weight: 700;
			color: {theme_tokens["value_color"]};
			margin-top: 0.2rem;
		}}
		</style>
		""",
		unsafe_allow_html=True,
	)


def main():
	inject_styles(OCEAN_THEME)
	_, model_file, data_file = app_paths()

	st.markdown(
		"""
		<section class="hero">
			<div class="hero-header">
				<h1>Salary Decider</h1>
				<span class="hero-chip">Enterprise Salary Recommendation</span>
			</div>
			<p>Use your trained model to decide fair employee salary bands based on role, experience, location, and employment context.</p>
			<div class="hero-grid">
				<div class="hero-mini">
					<div class="hero-mini-label">Decision Mode</div>
					<div class="hero-mini-value">Single Employee</div>
				</div>
				<div class="hero-mini">
					<div class="hero-mini-label">Model Type</div>
					<div class="hero-mini-value">Regression Pipeline</div>
				</div>
				<div class="hero-mini">
					<div class="hero-mini-label">Output</div>
					<div class="hero-mini-value">Recommended Salary</div>
				</div>
			</div>
		</section>
		""",
		unsafe_allow_html=True,
	)

	if not os.path.exists(model_file):
		st.error(f"Model file not found: {model_file}")
		st.info("Run training first: python scripts/modeltraining.py")
		return

	if not os.path.exists(data_file):
		st.error(f"Reference dataset not found: {data_file}")
		return

	model = get_model(model_file)
	ref_df = get_reference_data(data_file)

	features = [c for c in ref_df.columns if c != "salary"]
	last_input_df = pd.DataFrame(columns=features)
	metadata = {
		"status": "no_prediction_yet",
		"generated_at": pd.Timestamp.now().isoformat(),
	}

	col_form, col_result = st.columns([1.1, 0.9], gap="large")

	with col_form:
		st.subheader("Input Profile")
		with st.form("prediction_form", clear_on_submit=False):
			work_year = int(ref_df["work_year"].median())

			c1, c2 = st.columns(2)
			with c1:
				user_input = {
					"work_year": st.number_input(
						"Work Year",
						min_value=int(ref_df["work_year"].min()),
						max_value=int(ref_df["work_year"].max()),
						value=work_year,
						step=1,
					),
					"job_title": st.selectbox("Job Title", sorted(ref_df["job_title"].dropna().unique())),
					"job_category": st.selectbox("Job Category", sorted(ref_df["job_category"].dropna().unique())),
					"employee_residence": st.selectbox(
						"Employee Residence", sorted(ref_df["employee_residence"].dropna().unique())
					),
				}

			with c2:
				user_input.update(
					{
						"experience_level": st.selectbox(
							"Experience Level", sorted(ref_df["experience_level"].dropna().unique())
						),
						"employment_type": st.selectbox(
							"Employment Type", sorted(ref_df["employment_type"].dropna().unique())
						),
						"work_setting": st.selectbox("Work Setting", sorted(ref_df["work_setting"].dropna().unique())),
						"company_location": st.selectbox(
							"Company Location", sorted(ref_df["company_location"].dropna().unique())
						),
						"company_size": st.selectbox("Company Size", sorted(ref_df["company_size"].dropna().unique())),
					}
				)

			submitted = st.form_submit_button("Get Recommended Salary", use_container_width=True)

	with col_result:
		st.subheader("Salary Recommendation")
		st.caption("Model source: artifacts/pipeline.pkl")

		if submitted:
			input_df = pd.DataFrame([user_input], columns=features)
			last_input_df = input_df.copy()
			pred_df = predict_model(model, data=input_df)

			prediction_cols = ["prediction_label", "Label", "prediction"]
			pred_col = next((c for c in prediction_cols if c in pred_df.columns), None)

			if pred_col is None:
				st.error("Prediction column not found in model output.")
				st.dataframe(pred_df, use_container_width=True)
				return

			salary_pred = float(pred_df.loc[0, pred_col])
			salary_pred_usd = salary_pred/93.55
			metadata["status"] = "predicted"
			metadata["recommended_salary"] = salary_pred
			metadata["currency"] = "USD"
			metadata["prediction_column"] = pred_col
			metadata["selected_role"] = user_input["job_title"]
			metadata["experience_level"] = user_input["experience_level"]
			metadata["employment_type"] = user_input["employment_type"]
			metadata["generated_at"] = pd.Timestamp.now().isoformat()
			st.markdown(
				f"""
				<div class="result-card">
					<div class="result-label">Recommended Salary</div>
					<div class="result-value">${salary_pred_usd:,.2f}</div>
				</div>
				""",
				unsafe_allow_html=True,
			)

			st.markdown("### Input Snapshot")
			st.dataframe(input_df, use_container_width=True)
		else:
			st.info("Fill the form and click Get Recommended Salary to see the estimate.")

	if not submitted:
		last_input_df = pd.DataFrame([user_input], columns=features)

	st.sidebar.header("Insights")
	st.sidebar.metric("Input Features", len(features))
	st.sidebar.metric("Job Titles", int(ref_df["job_title"].nunique()))
	if "recommended_salary" in metadata:
		st.sidebar.metric("Latest Recommendation", f"${metadata['recommended_salary']:,.0f}")
	else:
		st.sidebar.metric("Latest Recommendation", "Not generated")

	# st.sidebar.subheader("Current Profile")
	# st.sidebar.json(user_input)

	# st.sidebar.subheader("Prediction Metadata")
	# st.sidebar.json(metadata)

	st.sidebar.header("Downloads")
	st.sidebar.download_button(
		label="Download Input Snapshot (CSV)",
		data=last_input_df.to_csv(index=False),
		file_name="input_snapshot.csv",
		mime="text/csv",
	)
	st.sidebar.download_button(
		label="Download Metadata (JSON)",
		data=pd.Series(metadata).to_json(indent=2),
		file_name="dashboard_metadata.json",
		mime="application/json",
	)

	st.sidebar.subheader("How To Use")
	st.sidebar.markdown(
		"1. Complete the employee profile.\n"
		"2. Click Get Recommended Salary.\n"
		"3. Export input and metadata for records."
	)


if __name__ == "__main__":
	main()
