{{- define "framewise.labels" -}}
app.kubernetes.io/name: framewise
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "framewise.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end -}}

{{- define "framewise.appEnv" -}}
- name: DATABASE_URL
  valueFrom:
    secretKeyRef: {name: {{ .Values.databaseSecret }}, key: DATABASE_URL}
- {name: FRAMEWISE_API_CONFIG, value: /app/configs/api.yaml}
- {name: FRAMEWISE_WORKER_CONFIG, value: /app/configs/worker.yaml}
{{- end -}}

{{- define "framewise.volumes" -}}
- name: configs
  configMap: {name: {{ .Release.Name }}-config}
- name: uploads
  persistentVolumeClaim: {claimName: {{ .Release.Name }}-uploads}
{{- end -}}

{{- define "framewise.mounts" -}}
- {name: configs, mountPath: /app/configs, readOnly: true}
- {name: uploads, mountPath: /app/data/uploads}
{{- end -}}
