################################################################################
#   Copyright (c) 2019 AT&T Intellectual Property.                             #
#                                                                              #
#   Licensed under the Apache License, Version 2.0 (the "License");            #
#   you may not use this file except in compliance with the License.           #
#   You may obtain a copy of the License at                                    #
#                                                                              #
#       http://www.apache.org/licenses/LICENSE-2.0                             #
#                                                                              #
#   Unless required by applicable law or agreed to in writing, software        #
#   distributed under the License is distributed on an "AS IS" BASIS,          #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   #
#   See the License for the specific language governing permissions and        #
#   limitations under the License.                                             #
################################################################################
{{/* vim: set filetype=mustache: */}}
{{/*
Expand the name of the chart.
*/}}
{{- define "ricxapp.name" -}}
  {{- default .Chart.Name .Values.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "ricxapp.fullname" -}}
  {{- $name := ( include "ricxapp.name" . ) -}}
  {{- $fullname := ( printf "%s-%s" .Release.Namespace $name ) -}}
  {{- default $fullname .Values.fullname | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ricxapp.chart" -}}
  {{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ricxapp.namespace" -}}
  {{- default .Release.Namespace .Values.nsPrefix -}}
{{- end -}}

{{- define "ricxapp.servicename.rmr" -}}
  {{- $name := ( include "ricxapp.fullname" . ) -}}
  {{- printf "service-%s-rmr" $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ricxapp.servicename.http" -}}
  {{- $name := ( include "ricxapp.fullname" . ) -}}
  {{- printf "service-%s-http" $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ricxapp.configmapname" -}}
  {{- $name := ( include "ricxapp.fullname" . ) -}}
  {{- printf "configmap-%s" $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ricxapp.servicename" -}}
  {{- $name := ( include "ricxapp.fullname" . ) -}}
  {{- printf "service-%s" $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ricxapp.deploymentname" -}}
  {{- $name := ( include "ricxapp.fullname" . ) -}}
  {{- printf "deployment-%s" $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}


{{- define "ricxapp.containername" -}}
  {{- $name := ( include "ricxapp.fullname" . ) -}}
  {{- printf "container-%s" $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}


{{- define "ricxapp.imagepullsecret" -}}
  {{- $reponame := .repo -}}
  {{- $postfix := $reponame | replace "." "-" | replace ":" "-" | replace "/" "-" | trunc 63 | trimSuffix "-" -}}
  {{- printf "secret-%s" $postfix -}}
{{- end -}}
