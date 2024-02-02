helm upgrade --install \
	--namespace riab \
	--values ./helm-charts/deep-watch-rapp/values.yaml \
	deep-watch-rapp \
	./helm-charts/deep-watch-rapp/ && \
	sleep 20 && \
	kubectl wait pod -n riab --for=condition=Ready -l app=deep-watch-rapp --timeout=600s
