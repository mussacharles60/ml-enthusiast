

async function query(filename) {
	const data = fs.readFileSync(filename);
	const response = await fetch(
		"https://api-inference.huggingface.co/models/mussacharles60/sw_asr_2",
		{
			headers: { Authorization: "Bearer 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" },
			method: "POST",
			body: data,
		}
	);
	const result = await response.json();
	return result;
}

query("D:\\Projects\\ml-enthusiast\\voice\\mcv_swahili\\datasets\\audio\\test\\sw_test_0\\sw_test_0\\common_voice_sw_27729868.mp3").then((response) => {
	console.log(JSON.stringify(response));
});