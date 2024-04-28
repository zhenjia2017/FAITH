import json
import requests

class SutimeClient:
	def __init__(self, host="localhost", port="7779"):
		self.host = host
		self.port = port
		self.req = requests.Session()

	def sutime_annotation_normalization(self, string, reference_time = "2023-01-01"):
		params = {"string": string, "reference_time": reference_time}
		res = self._req("/annotation", params)
		json_string = res.content.decode("utf-8")
		annotations = json.loads(json_string)
		return annotations

	def sutime_annotation_normalization_multithreading(self, string_refers):
		params = {"string_refers": string_refers}
		res = self._req("/multithread", params)
		json_string = res.content.decode("utf-8")
		annotations = json.loads(json_string)
		return annotations

	def _req(self, action, json):
		return self.req.post(self.host + ":" + self.port + action, json=json)


"""
MAIN
"""
if __name__ == "__main__":
	sut = SutimeClient()
	string = "what was the current population of japan ?"
	reference_time = "2023-01-01"
	res = sut.sutime_annotation_normalization(string, reference_time)
	print(res)
